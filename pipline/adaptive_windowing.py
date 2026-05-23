import logging
import numpy as np
import torch
from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Tuple, Deque

from src.preprocessing.seg_preprocess import prepare_segment # Corrected import path for prepare_segment

logger = logging.getLogger(__name__)


# -- Data Structures ---------------------------------------------------------

@dataclass
class Beat:
    """Represents a single detected heartbeat and its analytical boundaries."""
    cp_abs: int
    bl_I:   Optional[int] = None
    br_I:   Optional[int] = None
    bl_II:  Optional[int] = None
    br_II:  Optional[int] = None

    def segment_I(self, signal: np.ndarray) -> Optional[np.ndarray]:
        if self.bl_I is None or self.br_I is None:
            return None
        return signal[max(0, self.bl_I) : min(len(signal), self.br_I)]

    def segment_II(self, signal: np.ndarray) -> Optional[np.ndarray]:
        if self.bl_II is None or self.br_II is None:
            return None
        return signal[max(0, self.bl_II) : min(len(signal), self.br_II)]


@dataclass
class WindowState:
    """Holds mutable state during Algorithm 2. Prevents global variables."""
    omega:   float
    s:       float
    wst:     int
    C_bar:   float
    cp_min:  float
    cp_max:  float
    s_delta: bool = False


# -- Helpers ---------------------------------------------------------------

def find_critical_point(segment: np.ndarray) -> int:
    """
    Finds the Local index of the critical point within a segment.
    CP is local max or min, whichever is furthest from the median.
    """
    if len(segment) == 0:
        return 0

    med      = np.median(segment)
    i_max    = int(np.argmax(segment))
    i_min    = int(np.argmin(segment))
    dist_max = abs(segment[i_max] - med)
    dist_min = abs(segment[i_min] - med)

    return i_max if dist_max >= dist_min else i_min


# -- Core Algorithm --------------------------------------------------------

class AdaptiveWindowSegmenter:
    """
    Adaptive ECG segmentation pipeline based on CNN probabilities.
    """

    def __init__(self, cnn_model: torch.nn.Module, window_cfg, signal_cfg, device: str = "cpu"):
        """
        Dependency Injection applied. Configs must be passed explicitly.
        """
        self.cnn = cnn_model
        self.cnn.eval()
        self.cnn.to(device)
        self.device = device

        self.WC = window_cfg
        self.SC = signal_cfg

        # Canh bao nghich ly hoc thuat (The eta_s Paradox)
        if self.WC.eta_s >= 1.0:
            logger.warning(
                f"CRITICAL: WC.eta_s is {self.WC.eta_s}. Paper says 3*eta_w, " \
                f"but this skips multiple beats! Recommend setting eta_s to < 1.0 (e.g., eta_w/3)."
            )

    # -- Public API --------------------------------------------------------

    def segment(self, signal: np.ndarray) -> List[Beat]:
        """Main execution pipeline."""
        if len(signal) < self.SC.fs:
            logger.warning("Signal is shorter than 1 second. Segmentation might fail.")

        state, cp_history, c_history = self._initialize(signal)
        beats = self._run_segmentation(signal, state, cp_history, c_history)
        self._fill_method_II_boundaries(beats, state.C_bar)

        return beats

    # -- Algorithm 1: Init -------------------------------------------------

    def _initialize(
        self,
        signal: np.ndarray
    ) -> Tuple[WindowState, List[int], Deque[float]]:
        """
        Algorithm 1: Dry run on first K beats to warm up C_bar.
        """
        fs = self.SC.fs
        K  = self.WC.K
        WC = self.WC

        omega_0 = int(0.5 * fs)
        q = 0
        max_retries = 5

        while q < max_retries:
            omega = float(omega_0)
            s     = float(WC.eta_s * omega)
            wst   = 0
            j     = 0

            cp_hist: List[int] = []
            # O(1) operations with deque instead of list.pop(0)
            c_hist: Deque[float] = deque(maxlen=K)

            while j < K:
                end = wst + int(omega)
                if end > len(signal):
                    break  # End of signal reached

                zetatemp = signal[wst:end]
                if self._cnn_predict(zetatemp) >= WC.p_b:
                    cp_local = find_critical_point(zetatemp)
                    cp_abs   = wst + cp_local

                    if cp_hist:
                        c_hist.append(float(cp_abs - cp_hist[-1]))

                    cp_hist.append(cp_abs)
                    wst = cp_abs + int(omega * WC.eta_of)
                    j  += 1

                    if c_hist:
                        C_bar = np.mean(c_hist)
                        omega = WC.eta_w * C_bar
                        s      = WC.eta_s * C_bar
                else:
                    wst += int(s)

            # --- Safety Checks for convergence ---
            if j >= K:
                # Perfect initialization
                C_bar = float(np.mean(c_hist))
                return self._create_state(C_bar, wst), cp_hist, c_hist

            elif wst >= len(signal):
                # Reached end of file before finding K beats
                if j >= 3:
                    logger.info(f"Signal too short for {K} beats. Initialized with {j} beats.")
                    C_bar = float(np.mean(c_hist))
                    return self._create_state(C_bar, wst), cp_hist, c_hist
                else:
                    raise ValueError(f"Signal contains insufficient valid beats (found {j}).")

            else:
                # Stuck in noisy segment, loosen constraints
                q += 1
                omega_0 = int(0.5 * fs * np.exp(0.01 * q))

        raise RuntimeError("Initialization failed to converge after maximum retries.")

    def _create_state(self, C_bar: float, wst: int) -> WindowState:
        """Helper to construct WindowState uniformly."""
        return WindowState(
            omega   = self.WC.eta_w * C_bar,
            s       = self.WC.eta_s * C_bar,
            wst     = wst,
            C_bar   = C_bar,
            cp_min  = self.WC.eta_cmin * C_bar,
            cp_max  = self.WC.eta_cmax * C_bar,
            s_delta = False,
        )

    # -- Algorithm 2: Segmentation -----------------------------------------

    def _run_segmentation(
        self,
        signal:     np.ndarray,
        state:      WindowState,
        cp_history: List[int],
        c_history:  Deque[float],
    ) -> List[Beat]:
        """
        Algorithm 2: Dynamic tracking of beats and parameters.
        """
        WC = self.WC
        N  = len(signal)
        beats: List[Beat] = []

        while state.wst < N:
            end = min(state.wst + int(state.omega), N)
            zetatemp = signal[state.wst : end]

            # Avoid processing garbage tail data
            if len(zetatemp) < max(50, int(state.omega) // 4):
                break

            if self._cnn_predict(zetatemp) < WC.p_b:
                # No valid beat found, step forward
                step = WC.eta_sof * state.omega if state.s_delta else state.s
                state.wst += max(1, int(step))
                continue

            # --- Beat Found ---
            cp_local = find_critical_point(zetatemp)
            cp_abs   = state.wst + cp_local
            c_tilde  = float(cp_abs - cp_history[-1]) if cp_history else state.C_bar

            # Validate the beat interval
            if state.cp_min <= c_tilde <= state.cp_max:
                # Valid distance
                state.s_delta = False
                c_history.append(c_tilde)

                # Update dynamic parameters
                state.C_bar  = float(np.mean(c_history))
                state.omega  = WC.eta_w * state.C_bar
                state.s      = WC.eta_s * state.C_bar
                state.cp_min = WC.eta_cmin * state.C_bar
                state.cp_max = WC.eta_cmax * state.C_bar

            elif c_tilde < state.cp_min:
                # False positive / Noise close to last beat -> Small step
                state.s_delta = True
                state.wst     = cp_abs + max(1, int(state.omega * WC.eta_sof))
                continue
            else:
                # Abnormally far (missed a beat perhaps). Accept it, but don't poison C_bar
                state.s_delta = False

            # Calculate Method I Boundaries
            cp_history.append(cp_abs)
            bl_I = cp_abs - int(state.C_bar * WC.eta_delta * WC.eta_ar)
            br_I = cp_abs + int(state.C_bar * WC.eta_delta * (1 - WC.eta_ar))

            beats.append(Beat(cp_abs=cp_abs, bl_I=bl_I, br_I=br_I))

            # Move window past current beat
            state.wst = cp_abs + int(state.omega * WC.eta_of)

        return beats

    def _fill_method_II_boundaries(self, beats: List[Beat], default_c: float):
        """
        Retroactive boundaries for Method II.
        BUG FIXED: Now processes the final beat properly.
        """
        if len(beats) < 2:
            return

        WC = self.WC
        for idx in range(1, len(beats)):
            prev_beat = beats[idx - 1]
            curr_beat = beats[idx]

            c_prev  = curr_beat.cp_abs - prev_beat.cp_abs
            c_pprev = (prev_beat.cp_abs - beats[idx - 2].cp_abs) if idx >= 2 else c_prev

            prev_beat.bl_II = prev_beat.cp_abs - int(c_pprev * WC.eta_l)
            prev_beat.br_II = prev_beat.cp_abs + int(c_prev  * WC.eta_r)

        # Ensure the last beat is not left hanging
        last_beat = beats[-1]
        c_last = (last_beat.cp_abs - beats[-2].cp_abs) if len(beats) > 1 else default_c

        last_beat.bl_II = last_beat.cp_abs - int(c_last * WC.eta_l)
        last_beat.br_II = last_beat.cp_abs + int(c_last * WC.eta_r)

    # -- Internal CNN Interface --------------------------------------------

    def _cnn_predict(self, segment: np.ndarray) -> float:
        """Isolated CNN logic to prevent littering main algorithms with PyTorch."""
        x = prepare_segment(segment, M=self.SC.M, return_tensor=True)
        x = x.unsqueeze(0).to(self.device)  # [B=1, C=1, L=512]

        with torch.no_grad():
            p_vb = self.cnn.predict_proba(x)

        return p_vb.item()
