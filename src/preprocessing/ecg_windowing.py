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
3. AdaptiveWindowing    cắt theo RR-interval [R_i:R_{i+1}] + PPLD nonlinear_warp()

Cấu trúc sinh lý trong window [R_i : R_{i+1}] (4 zones, R ở index 0):
    ┌──────────┬────────────┬──────────┬────────────────┐
    │  QRS     │  ST+T wave │ diastole │  P + PR        │
    │ (Zone 0) │  (Zone 1)  │ (Zone 2) │  (Zone 3)      │
    └──────────┴────────────┴──────────┴────────────────┘
    0        s_off       t_end     p_start             L_i
    ↑ R@0                                               ↑ R_{i+1}

Ràng buộc PPLD:
    Zone 0 (QRS)  : BẤT BIẾN (< 120ms, cấu trúc điện học cứng)
    Zone 1 (T)    : co giãn theo Fridericia / Framingham
    Zone 2,3      : elastic – tuyến tính tỷ lệ (diastole + P + PR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from omegaconf import DictConfig

try:
    import wfdb
    import wfdb.processing as wfdb_proc
    _WFDB_AVAILABLE = True
except ImportError:
    _WFDB_AVAILABLE = False

logger = logging.getLogger(__name__)

Formula = Literal["fridericia", "framingham"]


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
    signals: np.ndarray,
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
                continue
            wins.append(signals[start:end])
        if not wins:
            return np.empty((0, self.seq_len, leads), dtype=np.float32)
        return np.asarray(wins, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# Strategy 3 — AdaptiveWindowing + PPLD
# ══════════════════════════════════════════════════════════════

# ── 3a. Kiểu dữ liệu ranh giới ────────────────────────────────────────────

@dataclass
class RRBoundaries:
    """
    Ranh giới 4 zones trong window [R_i : R_{i+1}].

    R-peak nằm ở index 0 (đầu window).
    Zones tính từ index 0:
        Zone 0 – QRS      : [0,        s_offset)  ← BẤT BIẾN
        Zone 1 – ST+T     : [s_offset, t_end)      ← Fridericia / Framingham
        Zone 2 – diastole : [t_end,    p_start)    ← elastic
        Zone 3 – P+PR     : [p_start,  window_len) ← elastic (P ≤ 120ms)

    Attributes
    ----------
    s_offset   : S-point / J-point (cuối QRS, đầu ST)
    t_end      : offset sóng T (cuối tái cực)
    p_start    : onset sóng P của beat KẾ TIẾP
    window_len : = R_{i+1} - R_i  (= RR interval samples)
    rr_ms      : khoảng RR tính bằng ms
    """
    s_offset:   int
    t_end:      int
    p_start:    int
    window_len: int
    rr_ms:      float

    # ── derived ──────────────────────────────────────────────
    @property
    def qrs_duration(self) -> int:          # Zone 0  (R ở index 0)
        return self.s_offset

    @property
    def t_duration(self) -> int:            # Zone 1
        return self.t_end - self.s_offset

    @property
    def qt_duration(self) -> int:           # QT = Zone0 + Zone1
        return self.t_end

    @property
    def diastole(self) -> int:              # Zone 2
        return self.p_start - self.t_end

    @property
    def p_pr_duration(self) -> int:         # Zone 3
        return self.window_len - self.p_start

    def breakpoints(self) -> List[int]:
        """5 breakpoints → 4 zones."""
        return [0, self.s_offset, self.t_end, self.p_start, self.window_len]

    def to_dict(self) -> dict:
        return dict(s_offset=self.s_offset, t_end=self.t_end,
                    p_start=self.p_start, window_len=self.window_len,
                    rr_ms=self.rr_ms)

    @staticmethod
    def from_dict(d: dict) -> "RRBoundaries":
        return RRBoundaries(**d)


# ── 3b. Định luật Repolarization ──────────────────────────────────────────

def _qtc(qt_ms: float, rr_ms: float, formula: Formula) -> float:
    """QT → QTc."""
    rr_s = rr_ms / 1000.0
    if formula == "fridericia":
        return qt_ms / (rr_s ** (1.0 / 3.0))
    return qt_ms + 154.0 * (1.0 - rr_s)       # framingham


def _qt_from_qtc(qtc_ms: float, rr_ms: float, formula: Formula) -> float:
    """QTc → QT tại RR đích."""
    rr_s = rr_ms / 1000.0
    if formula == "fridericia":
        return qtc_ms * (rr_s ** (1.0 / 3.0))
    return qtc_ms - 154.0 * (1.0 - rr_s)       # framingham


# ── 3c. Heuristic delineator cho RR window ────────────────────────────────

def delineate_rr_window(
    window_1d:   np.ndarray,        # 1 lead, shape (L_i,)
    rr_ms:       float,
    fs:          float,
    formula:     Formula  = "fridericia",
    qtc_ms:      float    = 400.0,  # assumed population QTc
    qrs_dur_ms:  float    = 90.0,
    p_dur_ms:    float    = 90.0,
    pr_ms:       float    = 160.0,
) -> RRBoundaries:
    """
    Ước lượng 4-zone boundaries trong window [R_i : R_{i+1}].

    R-peak ở index 0. Tất cả thông số có thể override bằng annotation.

    Chiến lược:
      1. QRS width: gradient zero-crossing ± search quanh index 0 (fallback: qrs_dur_ms)
      2. T end    : s_offset + QT_Fridericia(rr_ms) - QRS
      3. P start  : window_len - round((p_dur_ms + pr_ms) / 1000 * fs)
      4. diastole : phần còn lại giữa T và P
    """
    N   = len(window_1d)
    spm = fs / 1000.0

    def ms2smp(x: float) -> int:
        return max(1, round(x * spm))

    # ── Zone 0: QRS ──────────────────────────────────────────────────────
    s_offset = _detect_s_offset(window_1d, fs, qrs_dur_ms)

    # ── Zone 1: T wave (Fridericia / Framingham) ─────────────────────────
    qt_ms  = _qt_from_qtc(qtc_ms, rr_ms, formula)
    qt_ms  = max(200.0, min(qt_ms, 600.0))
    qt_smp = ms2smp(qt_ms)
    t_end  = min(N - 1, qt_smp)                 # QT bắt đầu từ index 0 (Q≈R)

    # ── Zone 3 (ngược từ cuối): P+PR ────────────────────────────────────
    p_pr_smp = ms2smp(p_dur_ms + pr_ms)
    p_start  = max(t_end + 1, N - p_pr_smp)

    # Clamp: diastole không âm
    if p_start <= t_end:
        p_start = min(t_end + 1, N - 1)

    return RRBoundaries(
        s_offset=s_offset, t_end=t_end,
        p_start=p_start,   window_len=N,
        rr_ms=rr_ms,
    )


def _detect_s_offset(
    window_1d:  np.ndarray,
    fs:         float,
    qrs_dur_ms: float = 90.0,
    search_ms:  float = 60.0,
) -> int:
    """
    Tìm S-offset bằng gradient threshold từ index 0 (R ở đầu).
    Fallback về qrs_dur_ms nếu không phát hiện được.
    """
    N          = len(window_1d)
    spm        = fs / 1000.0
    search_smp = max(2, round(search_ms * spm))
    max_qrs    = max(1, round(120.0    * spm))
    fallback   = min(max_qrs, max(1, round(qrs_dur_ms * spm)))

    if N < 4:
        return min(fallback, N - 1)

    grad       = np.gradient(window_1d.astype(np.float64))
    r_grad_abs = max(abs(grad[0]), 1e-12)
    threshold  = r_grad_abs * 0.05
    right_lim  = min(N - 1, search_smp)

    for i in range(1, right_lim):
        if abs(grad[i]) < threshold:
            # Clamp sinh lý: 40ms ≤ QRS ≤ 120ms
            return max(round(40.0 * spm), min(i, max_qrs))

    return fallback


# ── 3d. Lõi biến dạng ─────────────────────────────────────────────────────

def _linear_resample_1d(sig: np.ndarray, n: int) -> np.ndarray:
    """Nội suy tuyến tính 1-D → n samples."""
    if n == 0:
        return np.array([], dtype=sig.dtype)
    if len(sig) == 0:
        return np.zeros(n, dtype=sig.dtype)
    if len(sig) == n:
        return sig.copy()
    if len(sig) == 1:
        return np.full(n, sig[0], dtype=sig.dtype)
    x_old = np.linspace(0.0, 1.0, len(sig))
    x_new = np.linspace(0.0, 1.0, n)
    return interp1d(x_old, sig, kind="linear")(x_new).astype(sig.dtype)


def _monotone(bp: List[int], total: int) -> List[int]:
    """Ép breakpoints non-decreasing; endpoint = total."""
    r = [0]
    for v in bp[1:-1]:
        r.append(max(r[-1], int(v)))
    r.append(total)
    return r


def _piecewise_warp_1d(
    sig:     np.ndarray,    # (L,)
    src_bp:  List[int],
    dst_bp:  List[int],
) -> np.ndarray:
    """
    Biến dạng tuyến tính từng đoạn trên 1-D signal.
    src_bp[i]..src_bp[i+1] → dst_bp[i]..dst_bp[i+1].
    """
    assert len(src_bp) == len(dst_bp), "Số breakpoints không khớp"
    M   = dst_bp[-1]
    out = np.empty(M, dtype=sig.dtype)
    for i in range(len(src_bp) - 1):
        s0, s1 = src_bp[i], src_bp[i + 1]
        d0, d1 = dst_bp[i], dst_bp[i + 1]
        out[d0:d1] = _linear_resample_1d(sig[s0:s1], d1 - d0)
    return out


def _compute_dst_breakpoints(
    b:          RRBoundaries,
    target_len: int,
    fs:         float,
    formula:    Formula,
) -> Tuple[List[int], List[int]]:
    """
    Tính (src_bp, dst_bp) cho PPLD 4-zone.

    Chiến lược phân bổ target_len:
      Zone 0 (QRS)  : giữ nguyên số sample, clamp ≤ 120ms
      Zone 1 (T)    : scaled theo QTc correction về RR_std = 1000ms
      Zone 2,3      : tuyến tính tỷ lệ với budget còn lại
    """
    spm     = fs / 1000.0

    # ── Zone 0: QRS bất biến ────────────────────────────────────────────
    max_qrs = max(1, round(120.0 * spm))
    qrs_dst = min(b.qrs_duration, max_qrs)

    # ── Zone 1: QT correction ───────────────────────────────────────────
    # qt_ms_orig = QT tại RR hiện tại (QT = t_end vì R ở index 0)
    qt_ms_orig = b.qt_duration / spm
    rr_std_ms  = 1000.0
    qtc_ms     = _qtc(qt_ms_orig, b.rr_ms, formula)
    qt_tgt_ms  = _qt_from_qtc(qtc_ms, rr_std_ms, formula)
    qt_tgt_ms  = max(200.0, min(qt_tgt_ms, 600.0))
    qt_dst     = max(qrs_dst + 1, round(qt_tgt_ms * spm))
    t_dst      = qt_dst - qrs_dst    # = Zone 1 length

    # ── Zone 2+3: elastic budget ────────────────────────────────────────
    budget     = max(0, target_len - qt_dst)
    elast_src  = max(1, b.diastole + b.p_pr_duration)

    def alloc(x: int) -> int:
        return max(1, round(x / elast_src * budget))

    dia_dst   = alloc(b.diastole)
    p_pr_dst  = max(1, budget - dia_dst)

    # ── Build breakpoints ────────────────────────────────────────────────
    s_o   = qrs_dst
    t_e   = qrs_dst + t_dst
    p_s   = t_e + dia_dst
    end   = p_s + p_pr_dst

    # Chuẩn hoá tổng = target_len
    raw   = [0, s_o, t_e, p_s, end]
    scale = target_len / max(end, 1)
    if abs(scale - 1.0) > 0.01:
        raw = [0] + [min(target_len, round(v * scale)) for v in raw[1:]]
    raw[-1] = target_len

    src_bp = _monotone(b.breakpoints(), b.window_len)
    dst_bp = _monotone(raw,             target_len)

    return src_bp, dst_bp


# ── 3e. Warp & Inverse per-beat (multi-lead) ──────────────────────────────

def _warp_beat(
    beat:       np.ndarray,     # (L_i, leads)
    src_bp:     List[int],
    dst_bp:     List[int],
) -> np.ndarray:
    """
    Áp dụng CÙNG breakpoints lên tất cả leads.
    Trả về (target_len, leads) float32.
    """
    leads  = beat.shape[1]
    t_len  = dst_bp[-1]
    warped = np.empty((t_len, leads), dtype=np.float32)
    for l in range(leads):
        warped[:, l] = _piecewise_warp_1d(beat[:, l], src_bp, dst_bp)
    return warped


def _inverse_warp_beat(
    warped:     np.ndarray,     # (target_len, leads)
    src_bp:     List[int],      # dst_bp từ lúc warp (bây giờ là nguồn)
    dst_bp:     List[int],      # src_bp từ lúc warp (bây giờ là đích)
) -> np.ndarray:
    """Nghịch đảo _warp_beat: (target_len, leads) → (L_orig, leads)."""
    leads    = warped.shape[1]
    orig_len = dst_bp[-1]
    restored = np.empty((orig_len, leads), dtype=np.float32)
    for l in range(leads):
        restored[:, l] = _piecewise_warp_1d(warped[:, l], src_bp, dst_bp)
    return restored


# ── 3f. AdaptiveWindowing class ───────────────────────────────────────────

class AdaptiveWindowing:
    """
    Cắt theo RR-interval [R_i : R_{i+1}] rồi co giãn phi tuyến
    về target_len bằng PPLD (Physiological Piecewise Linear Deformation).

    Config:  windowing.method     = "adaptive"
             windowing.target_len = 400        # e.g. 400 @ 360Hz ≈ 1111ms
             windowing.formula    = "fridericia"   (optional)
             windowing.qtc_ms     = 400.0          (optional)
             windowing.detector_lead = 0           (optional)

    Ràng buộc sinh lý:
      - QRS duration bất biến (< 120ms)
      - T wave co giãn theo Fridericia / Framingham
      - diastole + P + PR: elastic, tuyến tính tỷ lệ
    """

    def __init__(
        self,
        fs:              float,
        target_len:      int   | None = None,
        formula:         Formula      = "fridericia",
        qtc_ms:          float        = 400.0,
        qrs_dur_ms:      float        = 90.0,
        p_dur_ms:        float        = 90.0,
        pr_ms:           float        = 160.0,
        detector_lead:   int          = 0,
    ):
        """
        Parameters
        ----------
        fs             : sampling frequency (Hz)
        target_len     : độ dài chuẩn đầu ra (samples); None → tự suy luận
        formula        : "fridericia" | "framingham"
        qtc_ms         : assumed population QTc (ms) cho heuristic delineation
        qrs_dur_ms     : assumed QRS duration (ms)
        p_dur_ms       : assumed P duration (ms)
        pr_ms          : assumed PR interval (ms)
        detector_lead  : lead dùng để phát hiện ranh giới sóng (int index)
        """
        self.fs             = fs
        self.target_len     = target_len
        self.formula        = formula
        self.qtc_ms         = qtc_ms
        self.qrs_dur_ms     = qrs_dur_ms
        self.p_dur_ms       = p_dur_ms
        self.pr_ms          = pr_ms
        self.detector_lead  = detector_lead

    # ── Public: nonlinear_warp (single beat) ──────────────────────────────

    def nonlinear_warp(
        self,
        beat:       np.ndarray,     # (L_i, leads) — length thay đổi theo RR
        target_len: int,
    ) -> np.ndarray:
        """
        Co giãn phi tuyến beat [R_i:R_{i+1}] → target_len bằng PPLD.

        4-zone piecewise warping theo ràng buộc sinh lý:
          - Zone 0 (QRS)   : sample count bất biến (< 120ms)
          - Zone 1 (T)     : QT corrected về RR_std=1000ms (Fridericia/Framingham)
          - Zone 2,3       : diastole + P+PR elastic

        Breakpoints được tính từ detector_lead; áp dụng đồng nhất mọi lead.

        Parameters
        ----------
        beat       : (L_i, leads) float32, window = signals[R_i : R_{i+1}]
        target_len : độ dài đầu ra (samples)

        Returns
        -------
        warped : (target_len, leads) float32
        """
        L_i, leads = beat.shape
        rr_ms      = L_i / self.fs * 1000.0

        # ── 1. Delineate từ detector_lead ────────────────────────────────
        det_lead = min(self.detector_lead, leads - 1)
        b = delineate_rr_window(
            window_1d  = beat[:, det_lead],
            rr_ms      = rr_ms,
            fs         = self.fs,
            formula    = self.formula,
            qtc_ms     = self.qtc_ms,
            qrs_dur_ms = self.qrs_dur_ms,
            p_dur_ms   = self.p_dur_ms,
            pr_ms      = self.pr_ms,
        )

        # ── 2. Tính breakpoints ──────────────────────────────────────────
        src_bp, dst_bp = _compute_dst_breakpoints(b, target_len, self.fs, self.formula)

        # ── 3. Warp tất cả leads với cùng breakpoints ────────────────────
        return _warp_beat(beat, src_bp, dst_bp)

    # ── Public: apply (batch) ─────────────────────────────────────────────

    def apply(
        self,
        signals:        np.ndarray,       # (T, leads)
        rpeak_indices:  np.ndarray,       # int64 array, sample indices
    ) -> np.ndarray:
        """
        Trích và warp toàn bộ RR-windows.

        Parameters
        ----------
        signals       : (T, leads) float32, đã qua bandpass + zscore
        rpeak_indices : sample indices của R-peaks (từ detect_rpeaks)

        Returns
        -------
        tensor : (N, target_len, leads) float32
                 N = len(rpeak_indices) - 1  (bỏ beat cuối không đủ RR)
        """
        if self.target_len is None:
            raise ValueError(
                "target_len chưa được đặt. "
                "Truyền target_len vào __init__ hoặc set windowing.target_len trong config."
            )
        T, leads = signals.shape
        wins     = []

        for i in range(len(rpeak_indices) - 1):
            start = int(rpeak_indices[i])
            end   = int(rpeak_indices[i + 1])

            # Bỏ beat bị cắt ở biên hoặc quá ngắn (< 200ms)
            if start < 0 or end > T or (end - start) < round(0.2 * self.fs):
                continue

            beat   = signals[start:end]     # (L_i, leads)
            warped = self.nonlinear_warp(beat, self.target_len)
            wins.append(warped)

        if not wins:
            return np.empty((0, self.target_len, leads), dtype=np.float32)
        return np.asarray(wins, dtype=np.float32)   # (N, target_len, leads)

    # ── Public: inverse_warp (single beat) ───────────────────────────────

    def inverse_warp(
        self,
        warped:     np.ndarray,     # (target_len, leads)
        rr_ms:      float,          # RR interval ms của beat gốc
    ) -> np.ndarray:
        """
        Nghịch đảo nonlinear_warp: khôi phục (L_orig, leads) từ (target_len, leads).

        Dùng trong postprocess sau khi model sinh ra beat ở target_len
        và cần ghép lại vào timeline gốc.

        Parameters
        ----------
        warped  : (target_len, leads) output từ model
        rr_ms   : R-R interval ms của beat đích (có thể khác RR gốc)

        Returns
        -------
        restored : (L_orig, leads), L_orig = round(rr_ms / 1000 * fs)
        """
        target_len = warped.shape[0]
        L_orig     = max(1, round(rr_ms / 1000.0 * self.fs))
        leads      = warped.shape[1]

        det_lead = min(self.detector_lead, leads - 1)

        # Delineate trong không gian warped (detector_lead)
        # RR_std = 1000ms vì warped đã được co giãn về RR_std=1000ms
        b_std = delineate_rr_window(
            window_1d  = warped[:, det_lead],
            rr_ms      = 1000.0,            # warped space = RR_std
            fs         = self.fs,
            formula    = self.formula,
            qtc_ms     = self.qtc_ms,
            qrs_dur_ms = self.qrs_dur_ms,
            p_dur_ms   = self.p_dur_ms,
            pr_ms      = self.pr_ms,
        )

        # Tính breakpoints theo RR đích thực
        b_target = RRBoundaries(
            s_offset   = b_std.s_offset,   # QRS không đổi
            t_end      = _compute_t_end_at_rr(b_std, rr_ms, self.fs, self.formula),
            p_start    = _compute_p_start_at_rr(b_std, rr_ms, self.fs),
            window_len = L_orig,
            rr_ms      = rr_ms,
        )

        # src = warped space, dst = target space (đảo ngược)
        src_bp_fwd, dst_bp_fwd = _compute_dst_breakpoints(
            b_std, target_len, self.fs, self.formula
        )
        # Inverse: swap src ↔ dst
        src_bp_inv, dst_bp_inv = _compute_dst_breakpoints(
            b_target, L_orig, self.fs, self.formula
        )
        # Warp từ warped-space breakpoints → orig-space breakpoints
        return _inverse_warp_beat(warped, src_bp_fwd, dst_bp_inv)


# ── Helpers cho inverse_warp ───────────────────────────────────────────────

def _compute_t_end_at_rr(
    b_std:   RRBoundaries,
    rr_ms:   float,
    fs:      float,
    formula: Formula,
) -> int:
    """Tính t_end tại rr_ms từ QTc trong warped space."""
    spm        = fs / 1000.0
    qt_std_ms  = b_std.qt_duration / spm
    qtc_ms     = _qtc(qt_std_ms, 1000.0, formula)   # QTc tại RR_std=1000ms
    qt_tgt_ms  = _qt_from_qtc(qtc_ms, rr_ms, formula)
    qt_tgt_ms  = max(200.0, min(qt_tgt_ms, 600.0))
    return max(b_std.s_offset + 1, round(qt_tgt_ms * spm))


def _compute_p_start_at_rr(
    b_std:  RRBoundaries,
    rr_ms:  float,
    fs:     float,
) -> int:
    """Tính p_start tỷ lệ với rr_ms."""
    L_orig  = max(1, round(rr_ms / 1000.0 * fs))
    ratio   = b_std.p_start / max(b_std.window_len, 1)
    return max(_compute_t_end_at_rr(b_std, rr_ms, fs, "fridericia") + 1,
               round(ratio * L_orig))


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
            seq_len = cfg_windowing.seq_len,
            stride  = cfg_windowing.stride,
        )

    elif method == "rpeak":
        return RPeakWindowing(
            pre_peak_ms  = cfg_windowing.pre_peak_ms,
            post_peak_ms = cfg_windowing.post_peak_ms,
            fs           = fs,
        )

    elif method == "adaptive":
        return AdaptiveWindowing(
            fs             = fs,
            target_len     = cfg_windowing.get("target_len",    None),
            formula        = cfg_windowing.get("formula",        "fridericia"),
            qtc_ms         = cfg_windowing.get("qtc_ms",         400.0),
            qrs_dur_ms     = cfg_windowing.get("qrs_dur_ms",     90.0),
            p_dur_ms       = cfg_windowing.get("p_dur_ms",       90.0),
            pr_ms          = cfg_windowing.get("pr_ms",          160.0),
            detector_lead  = cfg_windowing.get("detector_lead",  0),
        )

    else:
        raise ValueError(
            f"Unknown windowing method '{method}'. "
            "Choose: hard_fixed | rpeak | adaptive"
        )
