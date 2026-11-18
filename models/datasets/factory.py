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

from typing import Iterable, Tuple, Union, Iterator
import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch

from physicsnemo.utils.diffusion import InfiniteSampler
from physicsnemo.distributed import DistributedManager

from models.datasets import base
from models.datasets.aorc_legacy import AORCDownscalingDataset
from models.datasets.wrf_legacy import WRFDownscalingDataset
from models.datasets.era5_conus404_legacy import ERA5CONUS404DownscalingDataset
from models.datasets.physical import PhysicalPairDataset
from models.datasets.frozen_aorc import FrozenAORCPairDataset
from legacy.aorc.matched_packed_dataset import AlignedPairedDataset

known_datasets = {
    "aorc": AORCDownscalingDataset,
    "wrf": WRFDownscalingDataset,
    "era5_conus404": ERA5CONUS404DownscalingDataset,
    "physical_pair": PhysicalPairDataset,
    "frozen_aorc": FrozenAORCPairDataset,
    "paper_matched": AlignedPairedDataset,
}


class ChunkedInfiniteSampler(torch.utils.data.Sampler[int]):
    """Infinite sampler that preserves chunk locality while keeping randomness.

    Instead of shuffling individual samples globally, this sampler shuffles chunk
    order and (optionally) shuffles samples inside each chunk. This is useful for
    time-chunked zarr data where fully random access is expensive.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        chunk_size: int,
        rank: int = 0,
        num_replicas: int = 1,
        seed: int = 0,
        shuffle_chunks: bool = True,
        shuffle_within_chunk: bool = True,
        start_idx: int = 0,
    ):
        if not len(dataset) > 0:
            raise ValueError("Dataset must contain at least one item")
        if not chunk_size > 0:
            raise ValueError("chunk_size must be positive")
        if not num_replicas > 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be non-negative and less than num_replicas")

        super().__init__()
        self.dataset = dataset
        self.chunk_size = int(chunk_size)
        self.rank = rank
        self.num_replicas = num_replicas
        self.seed = int(seed)
        self.shuffle_chunks = bool(shuffle_chunks)
        self.shuffle_within_chunk = bool(shuffle_within_chunk)
        self.start_idx = int(start_idx)
        self._order_size = len(dataset)
        self._n_chunks = (self._order_size + self.chunk_size - 1) // self.chunk_size

    def _build_epoch_order(self, epoch: int) -> np.ndarray:
        rng = np.random.RandomState(self.seed + int(epoch))
        chunk_ids = np.arange(self._n_chunks, dtype=np.int64)
        if self.shuffle_chunks and chunk_ids.size > 1:
            rng.shuffle(chunk_ids)

        parts = []
        for cid in chunk_ids:
            lo = int(cid * self.chunk_size)
            hi = min(lo + self.chunk_size, self._order_size)
            chunk = np.arange(lo, hi, dtype=np.int64)
            if self.shuffle_within_chunk and chunk.size > 1:
                rng.shuffle(chunk)
            parts.append(chunk)

        return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)

    def __iter__(self) -> Iterator[int]:
        epoch = self.start_idx // self._order_size
        pos = self.start_idx % self._order_size
        global_pos = self.start_idx
        order = self._build_epoch_order(epoch)

        while True:
            if pos >= self._order_size:
                epoch += 1
                pos = 0
                order = self._build_epoch_order(epoch)

            if global_pos % self.num_replicas == self.rank:
                yield int(order[pos])

            pos += 1
            global_pos += 1


def register_dataset(dataset_spec: str) -> None:
    """
    Register a new dataset class from a file path specification.

    Parameters
    ----------
    dataset_spec : str
        String specification in the format "path_to_file.py::dataset_class"

    Raises
    ------
    ValueError
        If the dataset_spec format is invalid or if the file doesn't exist
    ImportError
        If the dataset class cannot be imported
    """
    if dataset_spec in known_datasets:
        return  # Dataset already registered
    try:
        file_path, class_name = dataset_spec.split("::")
    except ValueError:
        raise ValueError(
            "Invalid dataset specification. Expected format: "
            "'path_to_file.py::dataset_class'"
        )
    if class_name in known_datasets:
        return  # Dataset already registered

    # Convert to Path and validate
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"Dataset file not found: {file_path}")
    if not file_path.suffix == ".py":
        raise ValueError(f"Dataset file must be a Python file: {file_path}")

    # Import the module and get the class
    spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        dataset_class = getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Could not find dataset class '{class_name}' in {file_path}")

    # Register the dataset
    known_datasets[dataset_spec] = dataset_class
    return


def init_train_valid_datasets_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    validation_dataset_cfg: Union[dict, None] = None,
    validation: bool = True,
    sampler_start_idx: int = 0,
) -> Tuple[
    base.DownscalingDataset,
    Iterable,
    Union[base.DownscalingDataset, None],
    Union[Iterable, None],
]:
    """
    A wrapper function for managing the train-test split for the CWB dataset.

    Parameters:
    - dataset_cfg (dict): Configuration for the dataset.
    - dataloader_cfg (dict, optional): Configuration for the dataloader. Defaults to None.
    - batch_size (int): The number of samples in each batch of data. Defaults to 1.
    - seed (int): The random seed for dataset shuffling. Defaults to 0.
    - validation (bool): A flag to determine whether to create a validation dataset. Defaults to True.
    - sampler_start_idx (int): The initial index of the sampler to use for resuming training. Defaults to 0.

    Returns:
    - Tuple[base.DownscalingDataset, Iterable, Optional[base.DownscalingDataset], Optional[Iterable]]: A tuple containing the training dataset and iterator, and optionally the validation dataset and iterator if `validation` is True.
    """

    config = copy.deepcopy(dataset_cfg)
    (dataset, dataset_iter) = init_dataset_from_config(
        config,
        dataloader_cfg,
        batch_size=batch_size,
        seed=seed,
        sampler_start_idx=sampler_start_idx,
    )
    if validation:
        valid_dataset_cfg = copy.deepcopy(config)
        if validation_dataset_cfg:
            valid_dataset_cfg.update(validation_dataset_cfg)
        (valid_dataset, valid_dataset_iter) = init_dataset_from_config(
            valid_dataset_cfg, dataloader_cfg, batch_size=batch_size, seed=seed
        )
    else:
        valid_dataset = valid_dataset_iter = None

    return dataset, dataset_iter, valid_dataset, valid_dataset_iter


def init_dataset_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    sampler_start_idx: int = 0,
) -> Tuple[base.DownscalingDataset, Iterable]:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop("type", "cwb")
    if "validation" in dataset_cfg:
        # handled by init_train_valid_datasets_from_config
        del dataset_cfg["validation"]
    quality_policy = dataset_cfg.pop("quality_policy", "strict")
    dataset_init_func = known_datasets[dataset_type]

    dataset_obj = dataset_init_func(**dataset_cfg)
    dataset_obj.quality_policy = quality_policy
    if dataloader_cfg is None:
        dataloader_cfg = {}
    else:
        dataloader_cfg = copy.deepcopy(dataloader_cfg)

    sampler_mode = dataloader_cfg.pop("sampler_mode", "infinite")
    sampler_shuffle = bool(dataloader_cfg.pop("sampler_shuffle", True))
    sampler_window_size = float(dataloader_cfg.pop("sampler_window_size", 0.5))
    sampler_chunk_size = dataloader_cfg.pop("sampler_chunk_size", None)
    sampler_shuffle_within_chunk = bool(
        dataloader_cfg.pop("sampler_shuffle_within_chunk", True)
    )

    dist = DistributedManager()
    if sampler_mode == "chunked":
        if sampler_chunk_size is None:
            raise ValueError(
                "sampler_chunk_size must be set when sampler_mode='chunked'."
            )
        dataset_sampler = ChunkedInfiniteSampler(
            dataset=dataset_obj,
            chunk_size=int(sampler_chunk_size),
            rank=dist.rank,
            num_replicas=dist.world_size,
            seed=seed,
            shuffle_chunks=sampler_shuffle,
            shuffle_within_chunk=sampler_shuffle_within_chunk,
            start_idx=sampler_start_idx,
        )
    elif sampler_mode == "infinite":
        dataset_sampler = InfiniteSampler(
            dataset=dataset_obj,
            rank=dist.rank,
            num_replicas=dist.world_size,
            shuffle=sampler_shuffle,
            seed=seed,
            window_size=sampler_window_size,
            start_idx=sampler_start_idx,
        )
    else:
        raise ValueError(
            f"Unknown sampler_mode '{sampler_mode}'. Use 'infinite' or 'chunked'."
        )

    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_size,
            worker_init_fn=None,
            **dataloader_cfg,
        )
    )

    return (dataset_obj, dataset_iterator)
