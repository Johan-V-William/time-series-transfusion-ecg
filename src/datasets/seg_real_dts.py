import numpy as np
import torch
import wfdb
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

from .preprocessing.seg_preprocess   import BeatTriplet, PositiveAugmenter, NegativeAugmenter, prepare_segment, extract_main_beat


class ECGDataset(Dataset):
    """
    Builds augmented dataset from MIT-BIH records.

    Label convention:
        1 = VB  (Valid Beat / positive)
        0 = NVB (Non-Valid Beat / negative)

    Usage:
        ds = ECGDataset(record_paths=["data/mit-bih/100", ...])
        x, y = ds[0]   # x: Tensor[1, 512], y: int
    """

    def __init__(
        self,
        record_paths: List[str],
        M: int = 512,
        augment: bool = True,
        transform=None,
    ):
        self.M = M
        self.augment = augment
        self.transform = transform

        self.pos_aug = PositiveAugmenter()
        self.neg_aug = NegativeAugmenter()

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._build(record_paths)

    def _build(self, record_paths: List[str]):
        """Load records, extract triplets, augment."""
        for path in record_paths:
            signal, annotations = self._load_record(path)
            triplets = self._extract_triplets(signal, annotations)

            for triplet in triplets:
                if self.augment:
                    pos_segs = self.pos_aug(triplet, self.M)
                    neg_segs = self.neg_aug(triplet)
                else:
                    # No augment: just the main beat (positive) and one negative
                    beat_seg, bl, br = extract_main_beat(triplet) # Corrected import
                    pos_segs = [beat_seg]
                    neg_segs = [triplet.signal[:len(beat_seg)]]  # placeholder

                for seg in pos_segs:
                    self.samples.append((seg, 1))  # VB
                for seg in neg_segs:
                    self.samples.append((seg, 0))  # NVB

    def _load_record(self, path: str) -> Tuple[np.ndarray, dict]:
        """
        Load ECG signal + annotations using wfdb.
        Returns:
            signal: np.ndarray [n_samples, n_leads] -- use lead 0
            annotations: dict with 'sample' (cp positions) and 'symbol' (beat type)
        """
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'atr')

        # MLII co QRS ro nhat
        signal = record.p_signal[:, 0]

        # 3. LOC NHIEU (CRITICAL):
        non_beat_symbols = set(['+', '~', '|', '"', '!', 'x', ']', '['])

        valid_indices = [
            i for i, sym in enumerate(ann.symbol)
            if sym not in non_beat_symbols
        ]

        filtered_samples = ann.sample[valid_indices]
        filtered_symbols = np.array(ann.symbol)[valid_indices]

        return signal, {
            'sample': filtered_samples,
            'symbol': filtered_symbols
        }

    def _extract_triplets(self, signal: np.ndarray, annotations: dict) -> List[BeatTriplet]:
        """
        For each annotated beat j, extract the surrounding triplet
        (cp_{j-1}, cp_j, cp_{j+1}) as a BeatTriplet.
        Skip first and last beat (no full triplet available).
        """
        triplets = []
        positions = annotations['sample']
        symbols = annotations['symbol']

        # Thong thuong: 'N' (Normal), 'L' (Left bundle branch block), 'R' (Right bundle branch block)
        VALID_BEAT_SYMBOLS = {'N', 'L', 'R'}

        for j in range(1, len(positions) - 1):
            # Kiem tra xem nhip o giua (nhip chinh) co phai la VB khong
            if symbols[j] not in VALID_BEAT_SYMBOLS:
                continue  # Bo qua, khong dung nhip bat thuong lam tam sinh mau

            cp_left  = positions[j - 1]
            cp_main  = positions[j]
            cp_right = positions[j + 1]

            # Them margin 10% an toan hai dau
            margin = int(0.1 * (cp_right - cp_left))
            start = max(0, cp_left - margin)
            end   = min(len(signal), cp_right + margin)

            excerpt = signal[start:end]

            triplets.append(BeatTriplet(
                signal=excerpt,
                cp_left=cp_left - start,
                cp_main=cp_main - start,
                cp_right=cp_right - start,
            ))

        return triplets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seg, label = self.samples[idx]
        x = prepare_segment(seg, M=self.M, return_tensor=True)
        if self.transform:
            x = self.transform(x)
        return x, label


def build_kfold_loaders(
    all_records: List[str],
    n_folds: int = 10,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Record-wise k-fold split.
    CRITICAL: split by RECORD not by sample -> no leakage.

    Yields (fold_idx, train_loader, test_loader) for each fold.
    """
    import torch
    from torch.utils.data import DataLoader

    fold_size = len(all_records) // n_folds

    for k in range(n_folds):
        test_records  = all_records[k * fold_size : (k + 1) * fold_size]
        train_records = [r for r in all_records if r not in test_records]

        train_ds = ECGDataset(train_records, augment=True)
        test_ds  = ECGDataset(test_records,  augment=True)  # augment test too (eval all variants)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        )

        yield k, train_loader, test_loader
