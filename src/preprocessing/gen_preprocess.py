"""
Classes
───────
_BaseECG      shared: load record, build split, shuffle, run()
FixedECG      windowing.method = "hard_fixed"
AdaptiveECG   windowing.method = "rpeak" | "adaptive"

Public API
──────────
    build_ecg(cfg) -> (train: np.ndarray, test: np.ndarray)
"""

from __future__ import annotations

import os
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from ecg_windowing import (
    preprocess_record,
    detect_rpeaks,
    build_windowing,
    HardFixedWindowing,
    RPeakWindowing,
    AdaptiveWindowing,
)

try:
    import wfdb
    _WFDB_AVAILABLE = True
except ImportError:
    _WFDB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# Base — shared scaffolding
# ══════════════════════════════════════════════════════════════

class _BaseECG:
    """
    Subclass override duy nhất _window_record() để chọn strategy.
    Mọi thứ còn lại (load, split, shuffle, run) dùng chung.
    """

    def __init__(self, cfg: DictConfig):
        if not _WFDB_AVAILABLE:
            raise ImportError("pip install wfdb")
        self.cfg = cfg
        self.s   = cfg.ecg
        self.bf  = cfg.ecg.bandpass_filter

    # ── signal preprocessing ────────────────────────────────

    def _preprocess(self, signals: np.ndarray) -> np.ndarray:
        """(T, leads) raw → (T, leads) bandpassed + z-scored."""
        return preprocess_record(
            signals,
            lowcut=self.bf.lowcut,
            highcut=self.bf.highcut,
            fs=self.bf.fs,
            order=self.bf.order,
        )

    # ── windowing  (override in subclass) ───────────────────

    def _window_record(self, processed: np.ndarray) -> np.ndarray:
        """processed: (T, leads) → (N, seq_len, leads)"""
        raise NotImplementedError

    # ── build one split ─────────────────────────────────────

    def _build_split(self, record_ids: list, desc: str) -> np.ndarray:
        all_windows = []
        for rec_id in tqdm(record_ids, desc=desc):
            try:
                record    = wfdb.rdrecord(os.path.join(self.s.data_dir, str(rec_id)))
                processed = self._preprocess(record.p_signal)   # (T, leads)
                windows   = self._window_record(processed)      # (N, L, leads)
                if len(windows) > 0:
                    all_windows.append(windows)
            except Exception as exc:
                print(f"  [skip] {rec_id}: {exc}")
        if not all_windows:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(all_windows, axis=0)

    def _shuffle(self, data: np.ndarray) -> np.ndarray:
        return data[np.random.permutation(len(data))]

    # ── public ──────────────────────────────────────────────

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (train_windows, test_windows).
        Split cố định theo record list — không random.
        """
        cls = self.__class__.__name__

        print(f"[{cls}] TRAIN — {len(list(self.s.train_records))} records")
        train = self._build_split(list(self.s.train_records), f"{cls}/train")

        print(f"[{cls}] TEST  — {len(list(self.s.test_records))} records")
        test  = self._build_split(list(self.s.test_records),  f"{cls}/test")

        if self.s.shuffle:
            train = self._shuffle(train)
            test  = self._shuffle(test)

        print(f"[{cls}] Done — train={train.shape}  test={test.shape}")
        return train, test


# ══════════════════════════════════════════════════════════════
# FixedECG — windowing.method = "hard_fixed"
# ══════════════════════════════════════════════════════════════

class FixedECG(_BaseECG):
    """
    Sliding window cố định. Không cần R-peak.

    Config:
        ecg.windowing.method:   "hard_fixed"
        ecg.windowing.seq_len:  512
        ecg.windowing.stride:   512
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._windower = build_windowing(self.s.windowing, fs=self.bf.fs)
        assert isinstance(self._windower, HardFixedWindowing), \
            "FixedECG yêu cầu windowing.method = 'hard_fixed'"

    def _window_record(self, processed: np.ndarray) -> np.ndarray:
        return self._windower.apply(processed)


# ══════════════════════════════════════════════════════════════
# AdaptiveECG — windowing.method = "rpeak" | "adaptive"
# ══════════════════════════════════════════════════════════════

class AdaptiveECG(_BaseECG):
    """
    R-peak detection + windowing căn theo nhịp tim.

    "rpeak"    — cắt [R - pre_ms : R + post_ms], seq_len fixed
    "adaptive" — cắt theo RR-interval rồi nonlinear_warp() về target_len

    Config:
        ecg.windowing.method:        "rpeak" | "adaptive"
        # rpeak:
        ecg.windowing.pre_peak_ms:   192
        ecg.windowing.post_peak_ms:  512
        # adaptive:
        ecg.windowing.target_len:    null   ← điền sau
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._windower = build_windowing(self.s.windowing, fs=self.bf.fs)
        assert isinstance(self._windower, (RPeakWindowing, AdaptiveWindowing)), \
            "AdaptiveECG yêu cầu windowing.method = 'rpeak' hoặc 'adaptive'"

    def _window_record(self, processed: np.ndarray) -> np.ndarray:
        rpeak_indices = detect_rpeaks(processed[:, 0], fs=self.bf.fs)
        return self._windower.apply(processed, rpeak_indices)


# ══════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════

def build_ecg(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch đến FixedECG hoặc AdaptiveECG theo windowing.method.

    Returns
    -------
    (train_windows, test_windows)
        train: (N_train, seq_len, leads) float32
        test:  (N_test,  seq_len, leads) float32
    """
    method = cfg.ecg.windowing.method

    if method == "hard_fixed":
        return FixedECG(cfg).run()
    elif method in ("rpeak", "adaptive"):
        return AdaptiveECG(cfg).run()
    else:
        raise ValueError(
            f"Unknown windowing method '{method}'. "
            "Choose: hard_fixed | rpeak | adaptive"
        )
