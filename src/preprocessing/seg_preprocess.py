"""
src/preprocessing/seg_preprocess.py
====================================
Preprocessing pipeline tích hợp:
  1. prepare_segment()         – chuẩn hoá đoạn tín hiệu thô → tensor đầu vào CNN
  2. PPLDPreprocessor          – wrap/unwrap Beat objects (Method I & II)
     dùng trực tiếp từ AdaptiveWindowSegmenter output

Quan hệ với codebase:
  ┌─────────────────────┐      segment()       ┌──────────────────────────┐
  │AdaptiveWindowSegmenter│ ─────────────────→ │  List[Beat]              │
  └─────────────────────┘                      │  beat.bl_I / br_I        │
           ↑                                   │  beat.bl_II / br_II      │
    _cnn_predict()                             └───────────┬──────────────┘
           │                                               │
    prepare_segment()  ←── (đây)              PPLDPreprocessor ← (đây)
                                                           │
                                              ┌────────────▼─────────────┐
                                              │  tensor (N, L_std)       │
                                              │  → Model → generated     │
                                              │  → postprocess → Beat    │
                                              └──────────────────────────┘
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

Formula = Literal["fridericia", "framingham"]


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 – prepare_segment()
# Chuẩn hoá đoạn tín hiệu → tensor 1×M cho CNN (dùng trong _cnn_predict)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_segment(
    segment:        np.ndarray,
    M:              int   = 512,
    return_tensor:  bool  = True,
    normalize:      bool  = True,
    bandpass:       bool  = False,
    fs:             float = 360.0,
    lowcut:         float = 0.5,
    highcut:        float = 40.0,
) -> torch.Tensor | np.ndarray:
    """
    Chuẩn hoá segment thô → input chuẩn cho CNN beat classifier.

    Pipeline:
      1. Bandpass filter (tuỳ chọn)          – loại nhiễu đường cơ sở / HF
      2. Resample tuyến tính về M samples    – fixed-length input
      3. Z-score normalisation               – zero-mean, unit-variance

    Parameters
    ----------
    segment       : raw 1-D ECG window (bất kỳ độ dài)
    M             : target length (phải khớp với CNN input, default 512)
    return_tensor : True → torch.FloatTensor [1, M]; False → np.ndarray [M]
    normalize     : True → Z-score; False → chỉ resample
    bandpass      : True → áp dụng Butterworth bandpass trước resample
    fs            : sampling frequency (Hz), cần cho bandpass
    lowcut/highcut: bandpass cutoffs (Hz)

    Returns
    -------
    torch.FloatTensor shape [1, M]  (nếu return_tensor=True)
    np.ndarray       shape [M]      (nếu return_tensor=False)
    """
    seg = np.asarray(segment, dtype=np.float32)

    if len(seg) == 0:
        seg = np.zeros(M, dtype=np.float32)

    # ── 1. Bandpass (tuỳ chọn) ────────────────────────────────────────────
    if bandpass and len(seg) > 15:
        try:
            nyq = 0.5 * fs
            low  = max(1e-4, lowcut  / nyq)
            high = min(0.99,  highcut / nyq)
            b, a = butter(4, [low, high], btype="band")
            seg  = filtfilt(b, a, seg).astype(np.float32)
        except Exception:
            pass   # fallback: tiếp tục không filter

    # ── 2. Resample → M ──────────────────────────────────────────────────
    if len(seg) != M:
        seg = _linear_resample(seg, M)

    # ── 3. Z-score ────────────────────────────────────────────────────────
    if normalize:
        std = seg.std()
        if std > 1e-8:
            seg = (seg - seg.mean()) / std
        else:
            seg = seg - seg.mean()

    if return_tensor:
        # shape [1, M] = [C=1, L=M] để unsqueeze(0) → [B=1, C=1, L=M]
        return torch.from_numpy(seg).unsqueeze(0)
    return seg


def _linear_resample(sig: np.ndarray, n: int) -> np.ndarray:
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


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 – Kiểu dữ liệu ranh giới sóng (dùng chung Method I & II)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QRSBoundaries:
    """
    Ranh giới QRS trong Interval I (local index trong window).
    r_idx      : R-peak local index  (= beat.cp_abs - beat.bl_I)
    q_onset    : Q-onset local index
    s_offset   : S-offset / J-point local index
    window_len : len(window I)
    """
    r_idx:      int
    q_onset:    int
    s_offset:   int
    window_len: int

    @property
    def qrs_duration(self) -> int:
        return self.s_offset - self.q_onset

    def to_dict(self) -> dict:
        return dict(r_idx=self.r_idx, q_onset=self.q_onset,
                    s_offset=self.s_offset, window_len=self.window_len)

    @staticmethod
    def from_dict(d: dict) -> "QRSBoundaries":
        return QRSBoundaries(**d)


@dataclass
class BeatBoundaries:
    """
    Ranh giới 6-zone trong Interval II (local index trong window).

    Zones:
      [diastole_pre | P | PR | QRS | T | post_T]
      0           p_s  p_e  q_o  s_o  t_e   window_len
    """
    p_start:    int
    p_end:      int
    q_onset:    int
    r_idx:      int
    s_offset:   int
    t_end:      int
    window_len: int

    @property
    def diastole_pre(self) -> int:  return self.p_start
    @property
    def p_duration(self)   -> int:  return self.p_end   - self.p_start
    @property
    def pr_interval(self)  -> int:  return self.q_onset - self.p_end
    @property
    def qrs_duration(self) -> int:  return self.s_offset - self.q_onset
    @property
    def t_duration(self)   -> int:  return self.t_end   - self.s_offset
    @property
    def qt_duration(self)  -> int:  return self.t_end   - self.q_onset
    @property
    def post_t(self)       -> int:  return self.window_len - self.t_end

    def breakpoints(self) -> List[int]:
        return [0, self.p_start, self.p_end, self.q_onset,
                self.s_offset, self.t_end, self.window_len]

    def to_dict(self) -> dict:
        return dict(p_start=self.p_start, p_end=self.p_end,
                    q_onset=self.q_onset, r_idx=self.r_idx,
                    s_offset=self.s_offset, t_end=self.t_end,
                    window_len=self.window_len)

    @staticmethod
    def from_dict(d: dict) -> "BeatBoundaries":
        return BeatBoundaries(**d)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 – Heuristic delineators (fallback khi không có external delineator)
# ══════════════════════════════════════════════════════════════════════════════

def delineate_qrs(
    window:      np.ndarray,
    r_idx:       int,
    fs:          float = 360.0,
    search_ms:   float = 80.0,
) -> QRSBoundaries:
    """
    Phát hiện Q-onset / S-offset bằng gradient zero-crossing quanh R-peak.

    Parameters
    ----------
    window    : Interval I signal (local coords)
    r_idx     : R-peak local index
    fs        : sampling frequency
    search_ms : phạm vi tìm kiếm Q/S (± ms quanh R)
    """
    N          = len(window)
    search_smp = max(2, round(search_ms / 1000 * fs))
    grad       = np.gradient(window.astype(np.float64))
    r_grad_abs = abs(grad[min(r_idx, N-1)]) + 1e-12
    threshold  = r_grad_abs * 0.05

    # Q-onset: từ R đi về trái, tìm gradient ≈ 0
    left_lim = max(0, r_idx - search_smp)
    q_onset  = left_lim
    for i in range(min(r_idx, N-1), left_lim, -1):
        if abs(grad[i]) < threshold:
            q_onset = i
            break

    # S-offset: từ R đi về phải
    right_lim = min(N - 1, r_idx + search_smp)
    s_offset  = right_lim
    for i in range(min(r_idx, N-1), right_lim):
        if abs(grad[i]) < threshold:
            s_offset = i
            break

    # Clamp sinh lý: 40ms ≤ QRS ≤ 120ms
    min_qrs = max(1, round(40  / 1000 * fs))
    max_qrs = max(1, round(120 / 1000 * fs))
    qrs_dur = max(min_qrs, min(s_offset - q_onset, max_qrs))

    q_pre    = round(0.40 * qrs_dur)
    q_onset  = max(0,     r_idx - q_pre)
    s_offset = min(N - 1, q_onset + qrs_dur)

    return QRSBoundaries(r_idx=r_idx, q_onset=q_onset,
                         s_offset=s_offset, window_len=N)


def delineate_beat(
    window:      np.ndarray,
    r_idx:       int,
    rr_ms:       float,
    fs:          float = 360.0,
    qrs_dur_ms:  float = 90.0,
    p_dur_ms:    float = 90.0,
    pr_ms:       float = 160.0,
    qtc_ms:      float = 400.0,
    formula:     Formula = "fridericia",
) -> BeatBoundaries:
    """
    Ước lượng 6 ranh giới sóng trong Interval II từ sinh lý học.

    Tất cả thông số có giá trị mặc định bình thường (normal sinus);
    override bằng annotation thực nếu có.
    """
    N   = len(window)
    spm = fs / 1000.0   # samples per ms

    def ms(x): return max(1, round(x * spm))

    qrs_smp = min(ms(qrs_dur_ms), ms(120.0))
    p_smp   = min(ms(p_dur_ms),   ms(120.0))
    pr_smp  = ms(pr_ms)

    # QT dựa trên Fridericia / Framingham
    qt_ms   = _qt_from_qtc(qtc_ms, rr_ms, formula)
    qt_ms   = max(200.0, min(qt_ms, 600.0))
    qt_smp  = ms(qt_ms)
    t_smp   = max(1, qt_smp - qrs_smp)

    # QRS: anchor tại r_idx
    q_pre    = round(0.40 * qrs_smp)
    q_onset  = max(0,     r_idx - q_pre)
    s_offset = min(N - 1, q_onset + qrs_smp)

    # P: ngược về trước Q-onset
    pr_end   = q_onset
    pr_start = max(0, pr_end  - pr_smp)
    p_end    = pr_start
    p_start  = max(0, p_end   - p_smp)

    # T: tiến về sau S-offset
    t_end    = min(N, s_offset + t_smp)

    return BeatBoundaries(
        p_start=p_start, p_end=p_end,
        q_onset=q_onset, r_idx=r_idx,
        s_offset=s_offset, t_end=t_end,
        window_len=N,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 – Định luật Repolarization (Fridericia / Framingham)
# ══════════════════════════════════════════════════════════════════════════════

def _qtc(qt_ms: float, rr_ms: float, formula: Formula) -> float:
    rr_s = rr_ms / 1000.0
    if formula == "fridericia":
        return qt_ms / (rr_s ** (1.0 / 3.0))
    else:  # framingham
        return qt_ms + 154.0 * (1.0 - rr_s)


def _qt_from_qtc(qtc_ms: float, rr_ms: float, formula: Formula) -> float:
    rr_s = rr_ms / 1000.0
    if formula == "fridericia":
        return qtc_ms * (rr_s ** (1.0 / 3.0))
    else:
        return qtc_ms - 154.0 * (1.0 - rr_s)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5 – Lõi biến dạng tuyến tính từng phần
# ══════════════════════════════════════════════════════════════════════════════

def _piecewise_warp(
    signal:  np.ndarray,
    src_bp:  List[int],
    dst_bp:  List[int],
) -> np.ndarray:
    """
    Biến dạng từng đoạn: src_bp[i]..src_bp[i+1] → dst_bp[i]..dst_bp[i+1].
    Số breakpoints phải bằng nhau; src_bp[0]=dst_bp[0]=0.
    """
    assert len(src_bp) == len(dst_bp)
    M   = dst_bp[-1]
    out = np.empty(M, dtype=signal.dtype)
    for i in range(len(src_bp) - 1):
        s0, s1 = src_bp[i], src_bp[i + 1]
        d0, d1 = dst_bp[i], dst_bp[i + 1]
        out[d0:d1] = _linear_resample(signal[s0:s1], d1 - d0)
    return out


def _monotone(bp: List[int], total: int) -> List[int]:
    """Ép breakpoints non-decreasing; endpoint = total."""
    r = [0]
    for v in bp[1:-1]:
        r.append(max(r[-1], int(v)))
    r.append(total)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 6 – PPLDPreprocessor: lớp tích hợp chính
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PPLDConfig:
    """Cấu hình PPLD – truyền từ CFG.ppld hoặc tạo inline."""
    L_std_I:   int     = 200          # target length Method I (samples)
    L_std_II:  int     = 4096         # target length Method II (samples)
    r_ratio:   float   = 0.4          # R-peak anchor ratio (empirical 0.40)
    formula:   Formula = "fridericia" # QT correction formula
    fs:        float   = 360.0        # sampling frequency (Hz)
    qtc_ms:    float   = 400.0        # assumed QTc for heuristic delineation
    qrs_ms:    float   = 90.0         # assumed QRS duration ms
    p_ms:      float   = 90.0         # assumed P duration ms
    pr_ms:     float   = 160.0        # assumed PR interval ms


class PPLDPreprocessor:
    """
    Wrap / Unwrap ECG beats cho model training & inference.

    Tích hợp trực tiếp với AdaptiveWindowSegmenter output (List[Beat]).

    Ví dụ sử dụng
    -------------
    >>> ppld = PPLDPreprocessor(PPLDConfig(fs=CFG.signal.fs))
    >>> signal = rec.p_signal[:, 0]
    >>> beats  = segmenter.segment(signal)
    >>>
    >>> # Preprocess batch
    >>> batch_I,  metas_I  = ppld.preprocess_beats_I(signal, beats)
    >>> batch_II, metas_II = ppld.preprocess_beats_II(signal, beats)
    >>>
    >>> # ... model forward ...
    >>> gen_I  = model_I(batch_I)
    >>> gen_II = model_II(batch_II)
    >>>
    >>> # Postprocess
    >>> restored_I  = ppld.postprocess_beats_I(gen_I,  beats, metas_I)
    >>> restored_II = ppld.postprocess_beats_II(gen_II, beats, metas_II)
    """

    def __init__(self, cfg: PPLDConfig):
        self.cfg = cfg

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _rr_ms(self, beats, idx: int) -> float:
        """
        Tính RR interval ms cho beat[idx] từ danh sách beats.
        Dùng khoảng cách cp[idx] - cp[idx-1].
        Nếu là beat đầu tiên, ước lượng từ cp[1] - cp[0].
        """
        cfg = self.cfg
        if len(beats) < 2:
            return 800.0   # fallback 75 bpm

        if idx == 0:
            rr_smp = beats[1].cp_abs - beats[0].cp_abs
        else:
            rr_smp = beats[idx].cp_abs - beats[idx - 1].cp_abs

        return rr_smp / cfg.fs * 1000.0

    def _extract_I(self, signal: np.ndarray, beat) -> Optional[np.ndarray]:
        """Trích Interval I từ signal, clamp biên."""
        if beat.bl_I is None or beat.br_I is None:
            return None
        l = max(0, beat.bl_I)
        r = min(len(signal), beat.br_I)
        return signal[l:r] if r > l else None

    def _extract_II(self, signal: np.ndarray, beat) -> Optional[np.ndarray]:
        """Trích Interval II từ signal, clamp biên."""
        if beat.bl_II is None or beat.br_II is None:
            return None
        l = max(0, beat.bl_II)
        r = min(len(signal), beat.br_II)
        return signal[l:r] if r > l else None

    # ── Method I: public API ──────────────────────────────────────────────────

    def preprocess_beats_I(
        self,
        signal:     np.ndarray,
        beats:      list,                         # List[Beat]
        qrs_bounds: Optional[List[Optional[QRSBoundaries]]] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Preprocess tất cả beats → tensor (N, L_std_I).

        Parameters
        ----------
        signal     : full ECG signal (1-D)
        beats      : List[Beat] từ AdaptiveWindowSegmenter.segment()
        qrs_bounds : List ranh giới QRS (None per element → heuristic)

        Returns
        -------
        tensor     : np.ndarray (N, L_std_I)
        metas      : List[dict] để dùng trong postprocess_beats_I()
        """
        cfg    = self.cfg
        outs, metas = [], []

        for idx, beat in enumerate(beats):
            win = self._extract_I(signal, beat)
            if win is None or len(win) < 4:
                # Tạo zero-padding nếu window không hợp lệ
                logger.warning(f"Beat {idx}: Invalid Interval I – using zeros.")
                outs.append(np.zeros(cfg.L_std_I, dtype=np.float32))
                metas.append({"_invalid": True, "L_std_I": cfg.L_std_I,
                               "r_ratio": cfg.r_ratio, "fs": cfg.fs,
                               "qrs_b_std": QRSBoundaries(
                                   r_idx=round(cfg.r_ratio * cfg.L_std_I),
                                   q_onset=max(0, round(cfg.r_ratio*cfg.L_std_I)-10),
                                   s_offset=min(cfg.L_std_I-1, round(cfg.r_ratio*cfg.L_std_I)+15),
                                   window_len=cfg.L_std_I).to_dict(),
                               "qrs_b_orig": QRSBoundaries(0,0,1,1).to_dict()})
                continue

            # R-peak local trong window I
            r_local = beat.cp_abs - max(0, beat.bl_I)

            # Ranh giới QRS
            if qrs_bounds is not None and qrs_bounds[idx] is not None:
                qb = qrs_bounds[idx]
            else:
                qb = delineate_qrs(win, r_idx=r_local, fs=cfg.fs)

            warped, meta = _preprocess_I(win, qb, cfg.L_std_I, cfg.r_ratio, cfg.fs)
            outs.append(warped)
            metas.append(meta)

        tensor = np.stack(outs, axis=0)
        return tensor, metas

    def postprocess_beats_I(
        self,
        generated:  np.ndarray,          # (N, L_std_I)
        beats:      list,                 # List[Beat] – để lấy I_len đích
        metas:      List[dict],
    ) -> List[Tuple[np.ndarray, QRSBoundaries]]:
        """
        Postprocess tensor (N, L_std_I) → list của (restored_window, QRSBoundaries).
        Mỗi restored_window có độ dài = br_I - bl_I của beat tương ứng.
        """
        results = []
        for idx, (gen, beat, meta) in enumerate(zip(generated, beats, metas)):
            if meta.get("_invalid"):
                results.append((gen, QRSBoundaries.from_dict(meta["qrs_b_std"])))
                continue
            target_len = (beat.br_I - beat.bl_I
                          if beat.bl_I is not None and beat.br_I is not None
                          else self.cfg.L_std_I)
            restored, qb_out = _postprocess_I(gen, target_len, meta)
            results.append((restored, qb_out))
        return results

    # ── Method II: public API ─────────────────────────────────────────────────

    def preprocess_beats_II(
        self,
        signal:      np.ndarray,
        beats:       list,                         # List[Beat]
        beat_bounds: Optional[List[Optional[BeatBoundaries]]] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Preprocess tất cả beats → tensor (N, L_std_II).

        Parameters
        ----------
        signal      : full ECG signal
        beats       : List[Beat] từ AdaptiveWindowSegmenter
        beat_bounds : List ranh giới đầy đủ (None → heuristic)

        Returns
        -------
        tensor      : np.ndarray (N, L_std_II)
        metas       : List[dict] để dùng trong postprocess_beats_II()
        """
        cfg = self.cfg
        outs, metas = [], []

        for idx, beat in enumerate(beats):
            win = self._extract_II(signal, beat)
            if win is None or len(win) < 4:
                logger.warning(f"Beat {idx}: Invalid Interval II – using zeros.")
                outs.append(np.zeros(cfg.L_std_II, dtype=np.float32))
                metas.append({"_invalid": True, "L_std_II": cfg.L_std_II})
                continue

            rr_ms   = self._rr_ms(beats, idx)
            r_local = beat.cp_abs - max(0, beat.bl_II)

            if beat_bounds is not None and beat_bounds[idx] is not None:
                bb = beat_bounds[idx]
            else:
                bb = delineate_beat(win, r_idx=r_local, rr_ms=rr_ms,
                                    fs=cfg.fs, qrs_dur_ms=cfg.qrs_ms,
                                    p_dur_ms=cfg.p_ms, pr_ms=cfg.pr_ms,
                                    qtc_ms=cfg.qtc_ms, formula=cfg.formula)

            warped, meta = _preprocess_II(win, bb, rr_ms, cfg.L_std_II,
                                          cfg.r_ratio, cfg.formula, cfg.fs)
            outs.append(warped)
            metas.append(meta)

        tensor = np.stack(outs, axis=0)
        return tensor, metas

    def postprocess_beats_II(
        self,
        generated:  np.ndarray,          # (N, L_std_II)
        beats:      list,                 # List[Beat]
        metas:      List[dict],
    ) -> List[Tuple[np.ndarray, BeatBoundaries]]:
        """
        Postprocess tensor (N, L_std_II) → list (restored_window, BeatBoundaries).
        RR target = RR gốc của beat đó (bảo toàn độ dài sinh lý).
        """
        results = []
        for idx, (gen, beat, meta) in enumerate(zip(generated, beats, metas)):
            if meta.get("_invalid"):
                results.append((gen, None))
                continue
            rr_target = self._rr_ms(beats, idx)
            restored, bb_out = _postprocess_II(gen, rr_target, meta)
            results.append((restored, bb_out))
        return results

    # ── Convenience: cả hai method cùng lúc ──────────────────────────────────

    def preprocess_all(
        self,
        signal: np.ndarray,
        beats:  list,
    ) -> Tuple[np.ndarray, List[dict], np.ndarray, List[dict]]:
        """
        Trả về (tensor_I, metas_I, tensor_II, metas_II) trong một lần gọi.
        Tiện cho training loop.
        """
        t_I,  m_I  = self.preprocess_beats_I(signal, beats)
        t_II, m_II = self.preprocess_beats_II(signal, beats)
        return t_I, m_I, t_II, m_II


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 7 – Các hàm warp private (Method I)
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_I(
    window:   np.ndarray,
    qrs_b:    QRSBoundaries,
    L_std_I:  int,
    r_ratio:  float,
    fs:       float,
) -> Tuple[np.ndarray, dict]:
    """
    Co giãn Interval I → L_std_I.
    QRS core bất biến; Pre/Post co giãn tuyến tính; R anchored tại r_ratio×L_std.
    """
    N         = len(window)
    max_qrs   = max(1, round(120.0 / 1000.0 * fs))
    qrs_dst   = min(qrs_b.qrs_duration, max_qrs)

    # R-peak anchor
    r_dst     = round(r_ratio * L_std_I)
    q_to_r    = qrs_b.r_idx - qrs_b.q_onset
    q_to_r_sc = round(q_to_r / max(qrs_b.qrs_duration, 1) * qrs_dst)
    q_o       = max(0, r_dst - q_to_r_sc)
    s_o       = min(L_std_I, q_o + qrs_dst)

    src_bp = _monotone([0, qrs_b.q_onset, qrs_b.s_offset, N], N)
    dst_bp = _monotone([0, q_o,           s_o,            L_std_I], L_std_I)

    warped    = _piecewise_warp(window, src_bp, dst_bp)
    qrs_b_std = QRSBoundaries(r_idx=r_dst, q_onset=q_o, s_offset=s_o,
                               window_len=L_std_I)
    return warped, {
        "L_std_I":    L_std_I,
        "r_ratio":    r_ratio,
        "fs":         fs,
        "qrs_b_orig": qrs_b.to_dict(),
        "qrs_b_std":  qrs_b_std.to_dict(),
        "src_bp":     src_bp,
        "dst_bp":     dst_bp,
    }


def _postprocess_I(
    generated:    np.ndarray,
    target_len:   int,
    meta:         dict,
) -> Tuple[np.ndarray, QRSBoundaries]:
    """Nghịch đảo _preprocess_I."""
    L_std_I   = meta["L_std_I"]
    r_ratio   = meta["r_ratio"]
    qrs_b_std = QRSBoundaries.from_dict(meta["qrs_b_std"])
    qrs_dur   = qrs_b_std.qrs_duration

    r_out     = round(r_ratio * target_len)
    q_to_r    = qrs_b_std.r_idx - qrs_b_std.q_onset
    q_o       = max(0, r_out - q_to_r)
    s_o       = min(target_len, q_o + qrs_dur)

    src_bp = _monotone([0, qrs_b_std.q_onset, qrs_b_std.s_offset, L_std_I], L_std_I)
    dst_bp = _monotone([0, q_o,               s_o,                target_len], target_len)

    restored  = _piecewise_warp(generated, src_bp, dst_bp)
    qb_out    = QRSBoundaries(r_idx=r_out, q_onset=q_o, s_offset=s_o,
                               window_len=target_len)
    return restored, qb_out


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 8 – Các hàm warp private (Method II)
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_II(
    window:    np.ndarray,
    beat_b:    BeatBoundaries,
    rr_ms:     float,
    L_std_II:  int,
    r_ratio:   float,
    formula:   Formula,
    fs:        float,
) -> Tuple[np.ndarray, dict]:
    """
    Co giãn Interval II → L_std_II theo ràng buộc sinh lý:
      - QRS bất biến (samples)
      - T co giãn theo Fridericia / Framingham
      - P, PR, diastole, post-T elastic (tuyến tính)
      - R-peak anchored tại r_ratio × L_std_II
    """
    N   = len(window)
    spm = fs / 1000.0
    max_qrs = max(1, round(120.0 * spm))

    # ── QRS đích ─────────────────────────────────────────────────────────
    qrs_dst = min(beat_b.qrs_duration, max_qrs)

    # ── QT correction → T đích ───────────────────────────────────────────
    qt_ms_orig  = beat_b.qt_duration / spm
    rr_std_ms   = 1000.0
    qtc_ms      = _qtc(qt_ms_orig, rr_ms, formula)
    qt_tgt_ms   = _qt_from_qtc(qtc_ms, rr_std_ms, formula)
    qt_tgt_ms   = max(200.0, min(qt_tgt_ms, 600.0))
    qt_dst      = max(qrs_dst + 1, round(qt_tgt_ms * spm))
    t_dst       = qt_dst - qrs_dst

    # ── R-peak anchor ────────────────────────────────────────────────────
    r_dst       = round(r_ratio * L_std_II)
    q_to_r      = beat_b.r_idx - beat_b.q_onset
    q_to_r_sc   = round(q_to_r / max(beat_b.qrs_duration, 1) * qrs_dst)
    q_o         = max(0, r_dst - q_to_r_sc)
    s_o         = q_o + qrs_dst
    t_e         = s_o + t_dst

    # ── Elastic zones ────────────────────────────────────────────────────
    remaining    = max(0, L_std_II - qt_dst)
    elast_total  = max(1, beat_b.diastole_pre + beat_b.p_duration
                       + beat_b.pr_interval   + beat_b.post_t)

    def alloc(x: int) -> int:
        return max(1, round(x / elast_total * remaining))

    dia_dst  = alloc(beat_b.diastole_pre)
    p_dst    = min(alloc(beat_b.p_duration), max(1, round(120.0 * spm)))
    pr_dst   = alloc(beat_b.pr_interval)
    # post_t_dst: phần còn lại sau khi đặt Q-onset
    # Điều chỉnh q_o nếu elastic budget lệch khỏi r_anchor
    p_s  = dia_dst
    p_e  = p_s + p_dst
    # Ưu tiên r_anchor: tính lại pr_dst
    pr_dst = max(1, q_o - p_e)
    p_e    = max(p_s + 1, p_e)
    p_s    = max(0, p_e - p_dst)

    # Clamp t_e trong L_std_II
    t_e   = min(L_std_II - 1, t_e)
    s_o   = min(t_e - 1, s_o)
    q_o   = min(s_o, q_o)

    src_bp = _monotone(beat_b.breakpoints(), N)
    dst_bp = _monotone([0, p_s, p_e, q_o, s_o, t_e, L_std_II], L_std_II)

    warped     = _piecewise_warp(window, src_bp, dst_bp)
    beat_b_std = BeatBoundaries(
        p_start=dst_bp[1], p_end=dst_bp[2],
        q_onset=dst_bp[3], r_idx=r_dst,
        s_offset=dst_bp[4], t_end=dst_bp[5],
        window_len=L_std_II,
    )
    return warped, {
        "rr_ms":        rr_ms,
        "formula":      formula,
        "L_std_II":     L_std_II,
        "r_ratio":      r_ratio,
        "fs":           fs,
        "src_bp":       src_bp,
        "dst_bp":       dst_bp,
        "beat_b_orig":  beat_b.to_dict(),
        "beat_b_std":   beat_b_std.to_dict(),
    }


def _postprocess_II(
    generated:     np.ndarray,
    rr_target_ms:  float,
    meta:          dict,
) -> Tuple[np.ndarray, BeatBoundaries]:
    """Nghịch đảo _preprocess_II, scale về rr_target_ms."""
    L_std_II   = meta["L_std_II"]
    formula    = meta["formula"]
    r_ratio    = meta["r_ratio"]
    fs         = meta["fs"]
    beat_b_std = BeatBoundaries.from_dict(meta["beat_b_std"])
    spm        = fs / 1000.0

    out_len    = max(1, round(rr_target_ms / 1000.0 * fs))
    max_qrs    = max(1, round(120.0 * spm))
    qrs_dst    = min(beat_b_std.qrs_duration, max_qrs)

    # QT đích tại rr_target
    rr_std_ms  = 1000.0
    qt_std_ms  = beat_b_std.qt_duration / spm
    qtc_ms     = _qtc(qt_std_ms, rr_std_ms, formula)
    qt_tgt_ms  = _qt_from_qtc(qtc_ms, rr_target_ms, formula)
    qt_tgt_ms  = max(200.0, min(qt_tgt_ms, 600.0))
    qt_dst     = max(qrs_dst + 1, round(qt_tgt_ms * spm))
    t_dst      = qt_dst - qrs_dst

    r_out      = round(r_ratio * out_len)
    q_to_r     = beat_b_std.r_idx - beat_b_std.q_onset
    q_o        = max(0, r_out - q_to_r)
    s_o        = q_o + qrs_dst
    t_e        = min(out_len, s_o + t_dst)

    remaining  = max(0, out_len - qt_dst)
    elast_total = max(1, beat_b_std.diastole_pre + beat_b_std.p_duration
                      + beat_b_std.pr_interval   + beat_b_std.post_t)

    def alloc(x: int) -> int:
        return max(1, round(x / elast_total * remaining))

    p_dst   = min(alloc(beat_b_std.p_duration), max(1, round(120.0 * spm)))
    dia_dst = alloc(beat_b_std.diastole_pre)
    p_s     = dia_dst
    p_e     = p_s + p_dst
    pr_dst  = max(1, q_o - p_e)

    src_bp = _monotone(beat_b_std.breakpoints(), L_std_II)
    dst_bp = _monotone([0, p_s, p_e, q_o, s_o, t_e, out_len], out_len)

    restored   = _piecewise_warp(generated, src_bp, dst_bp)
    beat_b_out = BeatBoundaries(
        p_start=dst_bp[1], p_end=dst_bp[2],
        q_onset=dst_bp[3], r_idx=r_out,
        s_offset=dst_bp[4], t_end=dst_bp[5],
        window_len=out_len,
    )
    return restored, beat_b_out




# import numpy as np
# import torch
# from typing import List, Tuple, Union
# from dataclasses import dataclass
#
# def align_segment(v: np.ndarray, M: int = 512) -> np.ndarray:
#     '''Eq.(1): Center-pad signal to fixed length M using boundary values.
#     NOT zero-padding -- extends first/last sample outward.
#
#     Args:
#         v: ECG segment
#         M: target fixed length
#     Returns:
#         w: padded array of length M
#     '''
#     L = len(v)
#     assert L <= M, f"Segment length {L} exceeds M={M}"
#
#     pad_left = (M - L) // 2
#     pad_right = M - L - pad_left
#
#     w = np.concatenate([
#         np.full(pad_left,  v[0]),   # extend left with first value
#         v,
#         np.full(pad_right, v[-1]),  # extend right with last value
#     ])
#     assert len(w) == M
#     return w
#
#
# def minmax_normalize(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
#     '''Eq.(2): Normalize to [0, 1].
#     WARNING: applied per-segment, so amplitude info is lost.
#     '''
#     mn, mx = w.min(), w.max()
#     return (w - mn) / (mx - mn + eps)
#
#
# def prepare_segment(
#     v: np.ndarray,
#     M: int = 512,
#     return_tensor: bool = True,
# ) -> Union[np.ndarray, torch.Tensor]:
#     '''Full pipeline: align -> normalize -> optionally convert to tensor.
#     This is what gets fed into the CNN.
#
#     Returns shape: [1, M] (channel-first for Conv1D)
#     '''
#     # Ensure segment length does not exceed M
#     L = len(v)
#     if L > M:
#         # If segment is too long, take the middle M samples
#         start_idx = (L - M) // 2
#         v = v[start_idx : start_idx + M]
#         L = M # Update L to the new length, though not strictly needed here
#
#     w = align_segment(v, M)
#     x = minmax_normalize(w)
#
#     if return_tensor:
#         return torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, M]
#     return x
#
# @dataclass
# class BeatTriplet:
#     '''One annotated beat surrounded by its neighbors.'''
#     signal: np.ndarray   # raw ECG excerpt
#     cp_left:  int        # LOCAL index of left neighbor cp (cp_{j-1})
#     cp_main:  int        # LOCAL index of main beat cp (cp_j)
#     cp_right: int        # LOCAL index of right neighbor cp (cp_{j+1})
#
#
# def extract_main_beat(triplet: BeatTriplet) -> Tuple[np.ndarray, int, int]:
#     '''Extract main beat segment with boundaries:
#         bl = cp_main - 0.5 * d1
#         br = cp_main + 0.6 * d2
#     Returns: (segment, bl, br) -- all indices are local to triplet.signal
#     '''
#     d1 = triplet.cp_main - triplet.cp_left
#     d2 = triplet.cp_right - triplet.cp_main
#
#     # BUG 1 FIXED: cp_* are ALREADY local indices. No need to subtract offset.
#     bl = triplet.cp_main - int(0.5 * d1)
#     br = triplet.cp_main + int(0.6 * d2)
#
#     # Safety bounds to prevent out-of-index slicing
#     bl = max(0, bl)
#     br = min(len(triplet.signal), br)
#
#     return triplet.signal[bl:br], bl, br
#
#
# class PositiveAugmenter:
#     '''Creates 12 positive (valid heartbeat) samples per beat.
#     Fig.4: 1 main + 6 shift (+/-4%, +/-8%, +/-12% of s) + 5 trim
#     '''
#
#     SHIFT_FRACS = (0.04, 0.08, 0.12)   # applied left AND right
#     TRIM_FRACS  = (0.08, 0.16, 0.20, 0.24, 0.30)  # trimmed from RIGHT of window
#
#     def __call__(self, triplet: BeatTriplet, M: int = 512) -> List[np.ndarray]:
#         '''Returns list of 12 raw segments (before align+normalize).'''
#         beat, bl, br = extract_main_beat(triplet)
#         s = len(beat)
#         samples = []
#
#         # 1) Main beat (centred)
#         samples.append(triplet.signal[bl:br])
#
#         # 2) Shifted versions (left and right)
#         for frac in self.SHIFT_FRACS:
#             shift = int(frac * s)
#             # Left shift: window moves left -> beat appears right-of-centre
#             if bl - shift >= 0 and br - shift <= len(triplet.signal):
#                 samples.append(triplet.signal[bl - shift : br - shift])
#             # Right shift: window moves right -> beat appears left-of-centre
#             if bl + shift >= 0 and br + shift <= len(triplet.signal):
#                 samples.append(triplet.signal[bl + shift : br + shift])
#
#         # 3) Trimmed versions
#         # BUG 4 FIXED: Trim from the RIGHT end (decrease br) as per paper Fig.4
#         for frac in self.TRIM_FRACS:
#             trim = int(frac * s)
#             new_br = br - trim
#             if new_br > bl:
#                 samples.append(triplet.signal[bl:new_br])
#
#         # Guarantee exactly 12 (pad with main if augmentations fail at edges)
#         while len(samples) < 12:
#             samples.append(triplet.signal[bl:br])
#
#         return samples[:12]
#
#
# class NegativeAugmenter:
#     '''Creates 12 negative (non-valid) samples per beat.
#     '''
#     def __call__(self, triplet: BeatTriplet) -> List[np.ndarray]:
#         '''Returns list of 12 negative raw segments.'''
#         beat, bl, br = extract_main_beat(triplet)
#         s = len(beat)
#         samples = []
#
#         # Type 1: Extreme left shift (beats near leftmost corner)
#         for frac in (0.35, 0.45, 0.55):
#             shift = int(frac * s)
#             new_bl = max(0, bl - shift)
#             new_br = new_bl + s
#             if new_br <= len(triplet.signal):
#                 samples.append(triplet.signal[new_bl:new_br])
#
#         # Type 2: Extreme right shift (beats near rightmost corner)
#         for frac in (0.35, 0.45, 0.55):
#             shift = int(frac * s)
#             new_br = min(len(triplet.signal), br + shift)
#             new_bl = new_br - s
#             if new_bl >= 0:
#                 samples.append(triplet.signal[new_bl:new_br])
#
#         # Type 3: Two-cp segments (spans two beats -- clearly negative)
#         two_cp_span = triplet.signal[triplet.cp_left : triplet.cp_right]
#         if len(two_cp_span) > 0 and len(two_cp_span) <= 512:
#             samples.append(two_cp_span)
#
#         # Type 4: Heavy trim (almost no QRS visible - trim from left this time to ruin morphology)
#         for frac in (0.70, 0.80):
#             trim = int(frac * s)
#             new_bl = bl + trim
#             if new_bl < br:
#                 samples.append(triplet.signal[new_bl:br])
#
#         # Guarantee exactly 12 samples
#         while len(samples) < 12:
#             samples.append(triplet.signal[bl : bl + max(1, s // 2)])
#
#         return samples[:12]
