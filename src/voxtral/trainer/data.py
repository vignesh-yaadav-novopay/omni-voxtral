import os
import random
import typing

import numpy as np
import torch
import torch.utils.data as td

from .config import VoxtralTrainConfig


def get_npy_files(path: str) -> list[str]:
    npy_files: list[str] = []
    for root, _, files in os.walk(path, followlinks=True):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    return npy_files


def get_fake_item() -> dict[str, torch.Tensor]:
    return {"tokens": torch.randint(0, 1000, (220,))}


def get_item(file_path: str, max_seq_len: int | None = None) -> dict[str, torch.Tensor]:
    try:
        npy_data = np.load(file_path)
        item: dict[str, torch.Tensor] = {}

        item["tokens"] = torch.from_numpy(npy_data)

        if item["tokens"].dim() == 2:
            item["tokens"] = item["tokens"].squeeze()

        if max_seq_len is not None and item["tokens"].shape[0] > max_seq_len:
            item["tokens"] = item["tokens"][:max_seq_len]

        return item
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        # Generate a fake item as a fallback
        return get_fake_item()


class VoxtralDataset(td.IterableDataset):
    config: VoxtralTrainConfig
    data_step: int
    rank: int
    world_size: int
    file_paths: list[str]

    def __init__(self, config: VoxtralTrainConfig, split: str = "train", val_fraction: float = 0.1) -> None:
        super().__init__()
        self.config = config
        self.data_step = 0
        self.rank = config.rank
        self.world_size = config.world_size
        self.fake = config.fake
        self.overfit = config.overfit
        self.max_seq_len = config.max_seq_len

        if self.fake:
            self.file_paths = []
        else:
            all_files = get_npy_files(config.data_path)
            random.seed(config.seed)
            random.shuffle(all_files)
            # Deterministic train/val split
            val_count = max(1, int(len(all_files) * val_fraction))
            if split == "val":
                self.file_paths = all_files[:val_count]
            else:
                self.file_paths = all_files[val_count:]

        if not self.fake:
            print(f"Dataset split={split}: {len(self.file_paths)} files"
                  f" (val_fraction={val_fraction})")

    def __len__(self) -> int:
        if self.fake:
            return 100_000
        else:
            assert len(self.file_paths) > 0
            return len(self.file_paths)

    def __iter__(self) -> typing.Iterator[dict[str, torch.Tensor]]:
        worker_info = td.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        stride = num_workers * self.world_size
        offset = self.rank * num_workers + worker_id

        while True:
            self.data_step += stride
            idx = (offset + self.data_step) % len(self)
            if self.fake:
                yield get_fake_item()
            else:
                if self.overfit is not None:
                    idx = idx % self.overfit
                yield get_item(self.file_paths[idx], max_seq_len=self.max_seq_len)
