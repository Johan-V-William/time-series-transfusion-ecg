"""
ecg_windowing.py
================
Ba windowing strategies cho ECG pipeline.

Pipeline cố định:
    bandpass_filter  →  zscore_per_lead  →  windowing

Strategies
──────────
1. HardFixedWindowing   stride cố định, không cần R-peak
2. RPeakWindowing       căn window theo R-peak [R-pre : R+post]
3. AdaptiveWindowing    cắt theo RR-interval + nonlinear_warp()  ← TODO
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from omegaconf import DictConfig

try:
    import wfdb
    import wfdb.processing as wfdb_proc
    _WFDB_AVAILABLE = True
except ImportError:
    _WFDB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# Signal-level helpers  (dùng chung cho mọi strategy)
# ══════════════════════════════════════════════════════════════

def bandpass_filter(
    signal_1d: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal_1d)


def zscore_per_lead(signals: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(T, leads) → (T, leads) float32, mỗi lead chuẩn hoá độc lập."""
    out = np.empty_like(signals, dtype=np.float32)
    for lead in range(signals.shape[1]):
        s   = signals[:, lead]
        std = s.std()
        out[:, lead] = (s - s.mean()) / (std if std > eps else 1.0)
    return out


def preprocess_record(
    signals: np.ndarray,       # (T, leads) raw p_signal
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Bước 1+2: bandpass → zscore_per_lead. Trả về (T, leads) float32."""
    filtered = np.stack(
        [bandpass_filter(signals[:, l], lowcut, highcut, fs, order)
         for l in range(signals.shape[1])],
        axis=1,
    )
    return zscore_per_lead(filtered)


def detect_rpeaks(signal_1d: np.ndarray, fs: float) -> np.ndarray:
    """R-peak detection trên 1 lead (XQRS). Trả về sample indices (int64)."""
    if not _WFDB_AVAILABLE:
        raise ImportError("pip install wfdb")
    xqrs = wfdb_proc.XQRS(sig=signal_1d, fs=fs)
    xqrs.detect(verbose=False)
    return np.array(xqrs.qrs_inds, dtype=np.int64)


# ══════════════════════════════════════════════════════════════
# Strategy 1 — HardFixedWindowing
# ══════════════════════════════════════════════════════════════

class HardFixedWindowing:
    """
    Sliding window với seq_len và stride cố định.
    Không cần R-peak detection.

    Config:  windowing.method = "hard_fixed"
             windowing.seq_len, windowing.stride
    """

    def __init__(self, seq_len: int, stride: int):
        self.seq_len = seq_len
        self.stride  = stride

    def apply(self, signals: np.ndarray) -> np.ndarray:
        """(T, leads) → (N, seq_len, leads)"""
        T = len(signals)
        wins = [
            signals[i : i + self.seq_len]
            for i in range(0, T - self.seq_len + 1, self.stride)
        ]
        return np.asarray(wins, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# Strategy 2 — RPeakWindowing
# ══════════════════════════════════════════════════════════════

class RPeakWindowing:
    """
    Cắt window căn theo từng R-peak:
        beat = signal[R - pre_samples : R + post_samples]

    seq_len = pre_samples + post_samples  (fixed, tính từ ms + fs)

    Config:  windowing.method      = "rpeak"
             windowing.pre_peak_ms  = 192
             windowing.post_peak_ms = 512
    """

    def __init__(self, pre_peak_ms: float, post_peak_ms: float, fs: float):
        self.pre_samples  = round(pre_peak_ms  * 1e-3 * fs)
        self.post_samples = round(post_peak_ms * 1e-3 * fs)
        self.seq_len      = self.pre_samples + self.post_samples
        self.fs           = fs

    def apply(self, signals: np.ndarray, rpeak_indices: np.ndarray) -> np.ndarray:
        """(T, leads) + R-peak indices → (N, seq_len, leads)"""
        T, leads = signals.shape
        wins = []
        for r in rpeak_indices:
            start, end = r - self.pre_samples, r + self.post_samples
            if start < 0 or end > T:
                continue          # bỏ beat bị cắt ở biên
            wins.append(signals[start:end])
        if not wins:
            return np.empty((0, self.seq_len, leads), dtype=np.float32)
        return np.asarray(wins, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# Strategy 3 — AdaptiveWindowing  (TODO)
# ══════════════════════════════════════════════════════════════

class AdaptiveWindowing:
    """
    Cắt theo RR-interval [R_i : R_{i+1}] rồi co giãn phi tuyến
    về target_len bằng nonlinear_warp().

    Config:  windowing.method     = "adaptive"
             windowing.target_len = null   ← điền sau

    *** nonlinear_warp() cần bạn hiện thực bên dưới ***
    """

    def __init__(self, fs: float, target_len: int | None = None):
        self.fs         = fs
        self.target_len = target_len

    # ------------------------------------------------------------------
    # TODO — điền thuật toán co giãn phi tuyến của bạn vào đây
    # ------------------------------------------------------------------
    def nonlinear_warp(
        self,
        beat: np.ndarray,    # (L_i, leads) — length thay đổi theo RR
        target_len: int,
    ) -> np.ndarray:
        """
        Co giãn phi tuyến beat từ độ dài L_i → target_len.

        Returns
        -------
        warped : (target_len, leads) float32
        """
        raise NotImplementedError(
            "nonlinear_warp() chưa được hiện thực. "
            "Điền thuật toán vào đây."
        )
        # ── placeholder linear (xoá khi có thuật toán thật) ──────────
        # from scipy.interpolate import interp1d
        # x_old = np.linspace(0, 1, len(beat))
        # x_new = np.linspace(0, 1, target_len)
        # return np.stack(
        #     [interp1d(x_old, beat[:, l])(x_new) for l in range(beat.shape[1])],
        #     axis=1,
        # ).astype(np.float32)

    # ------------------------------------------------------------------

    def apply(self, signals: np.ndarray, rpeak_indices: np.ndarray) -> np.ndarray:
        """(T, leads) + R-peak indices → (N, target_len, leads)"""
        if self.target_len is None:
            raise NotImplementedError(
                "target_len chưa được đặt. "
                "Điền windowing.target_len vào config sau khi "
                "hiện thực nonlinear_warp()."
            )
        T, leads = signals.shape
        wins = []
        for i in range(len(rpeak_indices) - 1):
            start, end = rpeak_indices[i], rpeak_indices[i + 1]
            if start < 0 or end > T:
                continue
            warped = self.nonlinear_warp(signals[start:end], self.target_len)
            wins.append(warped)
        if not wins:
            return np.empty((0, self.target_len, leads), dtype=np.float32)
        return np.asarray(wins, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════

def build_windowing(
    cfg_windowing: DictConfig,
    fs: float,
) -> HardFixedWindowing | RPeakWindowing | AdaptiveWindowing:
    """Tạo windowing object từ config block ecg.windowing."""
    method = cfg_windowing.method

    if method == "hard_fixed":
        return HardFixedWindowing(
            seq_len=cfg_windowing.seq_len,
            stride=cfg_windowing.stride,
        )
    elif method == "rpeak":
        return RPeakWindowing(
            pre_peak_ms=cfg_windowing.pre_peak_ms,
            post_peak_ms=cfg_windowing.post_peak_ms,
            fs=fs,
        )
    elif method == "adaptive":
        return AdaptiveWindowing(
            fs=fs,
            target_len=cfg_windowing.get("target_len", None),
        )
    else:
        raise ValueError(
            f"Unknown windowing method '{method}'. "
            "Choose: hard_fixed | rpeak | adaptive"
        )
