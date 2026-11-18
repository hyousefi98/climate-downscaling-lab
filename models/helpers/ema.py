# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Exponential Moving Average (EMA) for Diffusion Model Weights.

EMA maintains a smoothed copy of model weights that is updated slowly during
training. This is standard practice for diffusion models because:

1. EMA weights are more stable (averages out noisy gradient updates)
2. EMA weights produce higher quality samples at inference time
3. EMA is robust to occasional bad batches or loss spikes

Usage:
    ema = EMA(model, decay=0.9999)

    # During training:
    for batch in dataloader:
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        ema.update(model)  # Update EMA after each step

    # For inference:
    ema.apply(model)  # Load EMA weights into model
    samples = model.sample(...)

    # Or save EMA weights separately:
    ema.save("ema_checkpoint.pt")
"""

import copy
from typing import Optional

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains shadow copies of model weights that are updated as:
        shadow = decay * shadow + (1 - decay) * current_weights

    Attributes
    ----------
    decay : float
        EMA decay rate. Higher values (e.g., 0.9999) make EMA update slower
        and provide more smoothing. Typical values: 0.999 to 0.9999.
    shadow : dict
        Dictionary containing the EMA weights (state_dict format).
    num_updates : int
        Number of update() calls. Used for warmup if enabled.
    warmup : bool
        If True, uses a warmup schedule where decay starts low and increases.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup: bool = True,
        warmup_power: float = 0.75,
    ):
        """
        Initialize EMA with a copy of the model weights.

        Parameters
        ----------
        model : nn.Module
            The model to track with EMA.
        decay : float, optional
            Target EMA decay rate. Default: 0.9999.
        warmup : bool, optional
            If True, use warmup schedule for decay. Default: True.
            During warmup, effective_decay = min(decay, (1 + num_updates) / (10 + num_updates))
            This prevents EMA from being too close to initial weights early in training.
        warmup_power : float, optional
            Power for warmup schedule. Default: 0.75.
        """
        self.decay = decay
        self.warmup = warmup
        self.warmup_power = warmup_power
        self.num_updates = 0

        # Create shadow copy of model weights
        self.shadow = {}
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name] = param.clone().detach()
            else:
                # Keep non-float params (e.g., batch norm running counts) as-is
                self.shadow[name] = param.clone()

    def _get_decay(self) -> float:
        """Get the current effective decay rate (with warmup if enabled)."""
        if self.warmup:
            # Warmup: decay starts low and increases to target
            # Formula from EDM paper / Karras et al.
            warmup_decay = (1 + self.num_updates) / (10 + self.num_updates)
            return min(self.decay, warmup_decay ** self.warmup_power)
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA weights with current model weights.

        Call this after each optimizer.step().

        Parameters
        ----------
        model : nn.Module
            The model with updated weights.
        """
        decay = self._get_decay()
        self.num_updates += 1

        for name, param in model.state_dict().items():
            if name in self.shadow:
                if param.dtype.is_floating_point:
                    # EMA update: shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name].mul_(decay).add_(param, alpha=1 - decay)
                else:
                    # Non-float params: just copy
                    self.shadow[name].copy_(param)

    def apply(self, model: nn.Module) -> None:
        """
        Load EMA weights into the model (for inference).

        Parameters
        ----------
        model : nn.Module
            The model to load EMA weights into.
        """
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module, backup: dict) -> None:
        """
        Restore model weights from a backup (after inference with EMA).

        Parameters
        ----------
        model : nn.Module
            The model to restore.
        backup : dict
            The backup state_dict (from model.state_dict() before apply()).
        """
        model.load_state_dict(backup)

    def state_dict(self) -> dict:
        """
        Get EMA state for checkpointing.

        Returns
        -------
        dict
            Dictionary containing EMA weights and metadata.
        """
        return {
            "shadow": self.shadow,
            "decay": self.decay,
            "num_updates": self.num_updates,
            "warmup": self.warmup,
            "warmup_power": self.warmup_power,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load EMA state from checkpoint.

        Parameters
        ----------
        state_dict : dict
            Dictionary from state_dict().
        """
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.warmup = state_dict.get("warmup", True)
        self.warmup_power = state_dict.get("warmup_power", 0.75)

    def save(self, path: str) -> None:
        """
        Save EMA weights to a file.

        Parameters
        ----------
        path : str
            Path to save the EMA checkpoint.
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, model: nn.Module) -> "EMA":
        """
        Load EMA from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to the EMA checkpoint.
        model : nn.Module
            The model (used to initialize EMA structure).

        Returns
        -------
        EMA
            Loaded EMA instance.
        """
        ema = cls(model)
        state_dict = torch.load(path, map_location="cpu")
        ema.load_state_dict(state_dict)
        return ema

    def copy_to_model_and_save(
        self,
        model: nn.Module,
        save_fn,
        path: str,
        **save_kwargs,
    ) -> None:
        """
        Temporarily apply EMA weights to model, save, then restore.

        This is useful for saving model checkpoints with EMA weights
        using the standard checkpoint saving mechanism.

        Parameters
        ----------
        model : nn.Module
            The model.
        save_fn : callable
            Function to call for saving (e.g., save_checkpoint).
        path : str
            Path to save to.
        **save_kwargs
            Additional arguments to pass to save_fn.
        """
        # Backup current weights
        backup = copy.deepcopy(model.state_dict())

        # Apply EMA weights
        self.apply(model)

        # Save with EMA weights
        save_fn(path=path, models=model, **save_kwargs)

        # Restore original weights
        self.restore(model, backup)
