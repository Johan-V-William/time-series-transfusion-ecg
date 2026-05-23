"""
load_data.py
============
Entry point → DataLoader.

    python load_data.py
    python load_data.py --config config/data_config.yaml --batch_size 128
"""

from __future__ import annotations

import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset_builder import build_datasets


def load_data(
    config_path: str = "config/data_config.yaml",
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    cfg = OmegaConf.load(config_path)
    train_ds, test_ds = build_datasets(cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/data_config.yaml")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train_loader, test_loader = load_data(args.config, args.batch_size)
    batch = next(iter(train_loader))
    print(f"Batch shape : {batch.shape}")          # (B, seq_len, leads)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test  batches: {len(test_loader)}")
