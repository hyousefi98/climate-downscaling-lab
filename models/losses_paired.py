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
Masked Loss Functions for CorrDiff Training.

These loss functions support ocean/land masking to exclude NaN regions
(ocean points) from the loss computation.

For AORC data, ocean regions have NaN values which cause problems in
standard loss functions. These masked versions:
1. Automatically detect NaN values in the target
2. Create a valid mask (True where data is valid, False for NaN/ocean)
3. Compute MSE only over valid points
4. Scale the loss appropriately
"""

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from physicsnemo.utils.patching import RandomPatching2D


class MaskedRegressionLoss:
    """
    Masked regression loss function for deterministic predictions.

    This loss computes MSE only over valid (non-NaN) pixels, allowing
    training on datasets with ocean/land masks where ocean is NaN.

    Note: this loss does not apply any reduction across batch/channel.
    """

    def __init__(self, mask: Optional[Tensor] = None):
        """
        Args:
            mask: Optional static mask tensor of shape (H, W) or (1, 1, H, W).
                  True = valid pixel, False = masked (excluded from loss).
                  If None, mask is derived from NaN values in target.
        """
        self.static_mask = mask

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        lead_time_label: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calculate masked regression loss.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model.
        img_clean : Tensor
            High-resolution target images of shape (B, C, H, W).
        img_lr : Tensor
            Low-resolution input images of shape (B, C, H, W).
        augment_pipe : callable, optional
            Data augmentation function.
        lead_time_label : Tensor, optional
            Lead time labels for temporal predictions.

        Returns
        -------
        Tensor
            Loss tensor of shape (B, C, H, W) with masked pixels set to 0.
        """
        # Augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        # Create zero input for regression
        zero_input = torch.zeros_like(y, device=img_clean.device)

        # Forward pass
        if lead_time_label is not None:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                augment_labels=augment_labels,
            )

        # Compute squared error
        squared_error = (D_yn - y) ** 2

        # Create mask from NaN values in target or use static mask
        if self.static_mask is not None:
            mask = self.static_mask.to(y.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
            mask = mask.expand_as(squared_error)
        else:
            # Derive mask from NaN values in target
            mask = ~torch.isnan(y)

        # Apply mask: set loss to 0 for masked (invalid) pixels
        # The loss for invalid pixels is 0, so they don't contribute
        masked_loss = torch.where(mask, squared_error, torch.zeros_like(squared_error))

        # Scale loss to account for masked pixels
        # Count valid pixels per sample/channel
        valid_count = mask.float().sum(dim=(-2, -1), keepdim=True)
        total_count = mask.shape[-2] * mask.shape[-1]

        # Scale factor: total / valid (so mean over valid pixels equals mean over all)
        scale = total_count / (valid_count + 1e-8)
        masked_loss = masked_loss * scale

        return masked_loss


class MaskedResidualLoss:
    """
    Masked residual loss for diffusion models with denoising score matching.

    This combines deterministic regression with denoising score matching,
    computing loss only over valid (non-NaN) pixels.

    IMPROVED VERSION with stability enhancements:
    - Optional loss clipping to prevent gradient explosion from outliers
    - Optional weight clipping to bound the EDM loss weighting
    - NaN/Inf protection in loss computation
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
        mask: Optional[Tensor] = None,
        residual_std: Optional[Tensor] = None,
        loss_clip_max: Optional[float] = None,
        weight_clip_max: float = 100.0,
    ):
        """
        Args:
            regression_net: Pre-trained regression network for residuals.
            P_mean: Mean for noise level computation.
            P_std: Std for noise level computation.
            sigma_data: Data standard deviation for weighting.
            hr_mean_conditioning: Use high-res mean for conditioning.
            mask: Optional static mask (H, W) or (1, 1, H, W).
            residual_std: Optional per-channel std for residual normalization.
            loss_clip_max: If set, clip per-pixel loss to this maximum value.
                          Helps prevent gradient explosion from outlier samples.
            weight_clip_max: Maximum value for loss weight (default: 100.0).
                            The EDM weight can be very large at low noise levels.
        """
        self.regression_net = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning
        self.static_mask = mask
        self.residual_std = residual_std
        self.y_mean = None
        self.loss_clip_max = loss_clip_max
        self.weight_clip_max = weight_clip_max

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        use_patch_grad_acc: bool = False,
    ) -> Tensor:
        """
        Calculate masked residual loss for diffusion.

        Parameters
        ----------
        net : torch.nn.Module
            Diffusion network.
        img_clean : Tensor
            High-resolution target (B, C, H, W).
        img_lr : Tensor
            Low-resolution input (B, C, H, W).
        patching : RandomPatching2D, optional
            Patching strategy for large images.
        lead_time_label : Tensor, optional
            Lead time labels.
        augment_pipe : callable, optional
            Data augmentation function.
        use_patch_grad_acc : bool
            Enable multi-iteration patching accumulation.

        Returns
        -------
        Tensor
            Masked loss tensor.
        """
        # Safety checks
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch: img_clean {img_clean.shape}, img_lr {img_lr.shape}"
            )

        # Augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # Compute regression mean (with gradient accumulation support)
        if use_patch_grad_acc:
            if self.y_mean is None:
                if lead_time_label is not None:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        lead_time_label=lead_time_label,
                        augment_labels=augment_labels,
                    )
                else:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        augment_labels=augment_labels,
                    )
                self.y_mean = y_mean
        else:
            if lead_time_label is not None:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    lead_time_label=lead_time_label,
                    augment_labels=augment_labels,
                )
            else:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    augment_labels=augment_labels,
                )
            self.y_mean = y_mean

        # Compute residual
        y = y - self.y_mean

        # Optional per-channel residual normalization
        if self.residual_std is not None:
            std = self.residual_std.to(y.device)
            if std.dim() == 1:
                std = std.view(1, -1, 1, 1)
            y = y / (std + 1e-8)

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)

        # Handle patching
        if patching:
            y_patched = patching.apply(input=y)
            y_lr_patched = patching.apply(input=y_lr, additional_input=img_lr)
            y = y_patched
            y_lr = y_lr_patched

        # Noise sampling from log-normal distribution
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # EDM loss weighting: weight = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        # This weight can be very large at small sigma (high SNR), causing instability
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # IMPROVED: Clip weight to prevent extreme values at low noise levels
        if self.weight_clip_max is not None:
            weight = torch.clamp(weight, max=self.weight_clip_max)

        # Add noise to create noisy latent
        latent = y + torch.randn_like(y) * sigma

        # Forward through diffusion network
        if lead_time_label is not None:
            D_yn = net(
                latent,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                latent,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                augment_labels=augment_labels,
            )

        # Compute weighted squared error
        raw_error = (D_yn - y) ** 2
        squared_error = weight * raw_error

        # IMPROVED: Handle NaN/Inf in model output (indicates numerical issues)
        if torch.isnan(squared_error).any() or torch.isinf(squared_error).any():
            # Replace NaN/Inf with zeros to prevent gradient explosion
            # This allows training to continue while flagging problematic samples
            squared_error = torch.nan_to_num(squared_error, nan=0.0, posinf=0.0, neginf=0.0)

        # IMPROVED: Optional per-pixel loss clipping to prevent outliers from dominating
        if self.loss_clip_max is not None:
            squared_error = torch.clamp(squared_error, max=self.loss_clip_max)

        # Create mask
        if self.static_mask is not None:
            mask = self.static_mask.to(y.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)

            # Handle patching case
            if patching:
                # Apply same patching to mask
                mask_expanded = mask.expand(batch_size, y.shape[1], -1, -1)
                mask = patching.apply(input=mask_expanded.float()) > 0.5
            else:
                mask = mask.expand_as(squared_error)
        else:
            # Derive from NaN in original y (before residual)
            # Note: y is now residual, so we check img_clean for NaN
            if patching:
                orig_mask = ~torch.isnan(img_clean)
                orig_mask_patched = patching.apply(input=orig_mask.float()) > 0.5
                mask = orig_mask_patched
            else:
                mask = ~torch.isnan(y_tot[:, : img_clean.shape[1], :, :])

        # Ensure boolean mask for torch.where
        if mask.dtype != torch.bool:
            mask = mask > 0.5

        # Apply mask
        masked_loss = torch.where(mask, squared_error, torch.zeros_like(squared_error))

        # Scale for masked pixels
        valid_count = mask.float().sum(dim=(-2, -1), keepdim=True)
        total_count = mask.shape[-2] * mask.shape[-1]
        scale = total_count / (valid_count + 1e-8)
        masked_loss = masked_loss * scale

        return masked_loss


def create_land_mask_from_data(data: Tensor, threshold: float = -1e10) -> Tensor:
    """
    Create a land mask from data by detecting ocean (NaN or sentinel values).

    Args:
        data: Tensor of shape (..., H, W)
        threshold: Values below this are considered ocean/invalid

    Returns:
        Boolean mask where True = land (valid), False = ocean (invalid)
    """
    # Check for NaN
    nan_mask = torch.isnan(data)

    # Check for sentinel values (very large negative)
    sentinel_mask = data < threshold

    # Land is where neither NaN nor sentinel
    land_mask = ~(nan_mask | sentinel_mask)

    # If multiple channels, all must be valid
    if land_mask.dim() > 2:
        # Reduce over channel dimension
        while land_mask.dim() > 2:
            land_mask = land_mask.all(dim=-3)

    return land_mask
