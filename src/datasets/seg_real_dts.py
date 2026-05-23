import numpy as np
import torch
import wfdb
from typing import List, Tuple
from torch.utils.data import Dataset

from src.preprocessing.seg_preprocess import (
    BeatTriplet,
    PositiveAugmenter,
    NegativeAugmenter,
    prepare_segment,
    extract_main_beat,
)


class ECGDataset(Dataset):
    def __init__(self, record_paths: List[str], M: int = 512, augment: bool = True, transform=None):
        self.M = M
        self.augment = augment
        self.transform = transform

        self.pos_aug = PositiveAugmenter()
        self.neg_aug = NegativeAugmenter()

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._build(record_paths)

    def _build(self, record_paths: List[str]):
        for path in record_paths:
            signal, annotations = self._load_record(path)
            triplets = self._extract_triplets(signal, annotations)

            for triplet in triplets:
                if self.augment:
                    pos_segs = self.pos_aug(triplet, self.M)
                    neg_segs = self.neg_aug(triplet)
                else:
                    beat_seg, _, _ = extract_main_beat(triplet)
                    pos_segs = [beat_seg]
                    neg_segs = [triplet.signal[: len(beat_seg)]]

                for seg in pos_segs:
                    self.samples.append((seg, 1))
                for seg in neg_segs:
                    self.samples.append((seg, 0))

    def _load_record(self, path: str) -> Tuple[np.ndarray, dict]:
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")

        signal = record.p_signal[:, 0]

        non_beat_symbols = set(["+", "~", "|", '"', "!", "x", "]", "["])
        valid_indices = [i for i, sym in enumerate(ann.symbol) if sym not in non_beat_symbols]

        filtered_samples = ann.sample[valid_indices]
        filtered_symbols = np.array(ann.symbol)[valid_indices]

        return signal, {"sample": filtered_samples, "symbol": filtered_symbols}

    def _extract_triplets(self, signal: np.ndarray, annotations: dict) -> List[BeatTriplet]:
        triplets = []
        positions = annotations["sample"]
        symbols = annotations["symbol"]

        valid_beat_symbols = {"N", "L", "R"}

        for j in range(1, len(positions) - 1):
            if symbols[j] not in valid_beat_symbols:
                continue

            cp_left = positions[j - 1]
            cp_main = positions[j]
            cp_right = positions[j + 1]

            margin = int(0.1 * (cp_right - cp_left))
            start = max(0, cp_left - margin)
            end = min(len(signal), cp_right + margin)

            excerpt = signal[start:end]

            triplets.append(
                BeatTriplet(
                    signal=excerpt,
                    cp_left=cp_left - start,
                    cp_main=cp_main - start,
                    cp_right=cp_right - start,
                )
            )

        return triplets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seg, label = self.samples[idx]
        x = prepare_segment(seg, M=self.M, return_tensor=True)
        if self.transform:
            x = self.transform(x)
        return x, label


def build_kfold_loaders(all_records: List[str], n_folds: int = 10, batch_size: int = 256, num_workers: int = 0):
    from torch.utils.data import DataLoader

    if n_folds <= 1:
        raise ValueError("n_folds must be >= 2")

    fold_size = max(1, len(all_records) // n_folds)

    for k in range(n_folds):
        start = k * fold_size
        end = min((k + 1) * fold_size, len(all_records))

        test_records = all_records[start:end]
        if not test_records:
            continue
        train_records = [r for r in all_records if r not in test_records]
        if not train_records:
            continue

        train_ds = ECGDataset(train_records, augment=True)
        test_ds = ECGDataset(test_records, augment=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        yield k, train_loader, test_loader
