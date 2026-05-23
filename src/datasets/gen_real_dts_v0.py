"""
dataset_builder.py
==================
Wrap (train_array, test_array) từ preprocessing thành PyTorch Datasets.

Usage
─────
    from dataset_builder import build_datasets

    train_ds, test_ds = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig

from src.preprocessing.gen_preprocess import build_ecg


# ══════════════════════════════════════════════════════════════
# Dataset wrapper
# ══════════════════════════════════════════════════════════════

class ECGDataset(Dataset):
    """
    Wraps (N, seq_len, leads) numpy array.
    __getitem__ trả về tensor (seq_len, leads).
    """

    def __init__(self, samples: np.ndarray):
        self.samples = torch.from_numpy(
            np.ascontiguousarray(samples)
        ).float()

    @property
    def num_samples(self)  -> int: return self.samples.shape[0]
    @property
    def seq_len(self)      -> int: return self.samples.shape[1]
    @property
    def num_leads(self)    -> int: return self.samples.shape[2]

    def __len__(self)              -> int:          return self.num_samples
    def __getitem__(self, idx: int) -> torch.Tensor: return self.samples[idx]

    def __repr__(self) -> str:
        return (f"ECGDataset(n={self.num_samples}, "
                f"seq_len={self.seq_len}, leads={self.num_leads})")


# ══════════════════════════════════════════════════════════════
# Builder
# ══════════════════════════════════════════════════════════════

def build_datasets(cfg: DictConfig) -> tuple[ECGDataset, ECGDataset]:
    """
    Chạy preprocessing → wrap thành ECGDataset.

    Split đã được thực hiện ở record level trong preprocessing
    nên không cần random split ở đây.

    Returns
    -------
    train_dataset, test_dataset : ECGDataset
    """
    train_arr, test_arr = build_ecg(cfg)

    train_ds = ECGDataset(train_arr)
    test_ds  = ECGDataset(test_arr)

    print(
        f"[dataset_builder] "
        f"method='{cfg.ecg.windowing.method}'  "
        f"train={train_ds}  test={test_ds}"
    )
    return train_ds, test_ds
