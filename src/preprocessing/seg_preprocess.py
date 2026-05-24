import numpy as np
import torch
from typing import List, Tuple, Union
from dataclasses import dataclass

def align_segment(v: np.ndarray, M: int = 512) -> np.ndarray:
    '''Eq.(1): Center-pad signal to fixed length M using boundary values.
    NOT zero-padding -- extends first/last sample outward.

    Args:
        v: ECG segment
        M: target fixed length
    Returns:
        w: padded array of length M
    '''
    L = len(v)
    assert L <= M, f"Segment length {L} exceeds M={M}"

    pad_left = (M - L) // 2
    pad_right = M - L - pad_left

    w = np.concatenate([
        np.full(pad_left,  v[0]),   # extend left with first value
        v,
        np.full(pad_right, v[-1]),  # extend right with last value
    ])
    assert len(w) == M
    return w


def minmax_normalize(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    '''Eq.(2): Normalize to [0, 1].
    WARNING: applied per-segment, so amplitude info is lost.
    '''
    mn, mx = w.min(), w.max()
    return (w - mn) / (mx - mn + eps)


def prepare_segment(
    v: np.ndarray,
    M: int = 512,
    return_tensor: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    '''Full pipeline: align -> normalize -> optionally convert to tensor.
    This is what gets fed into the CNN.

    Returns shape: [1, M] (channel-first for Conv1D)
    '''
    # Ensure segment length does not exceed M
    L = len(v)
    if L > M:
        # If segment is too long, take the middle M samples
        start_idx = (L - M) // 2
        v = v[start_idx : start_idx + M]
        L = M # Update L to the new length, though not strictly needed here

    w = align_segment(v, M)
    x = minmax_normalize(w)

    if return_tensor:
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, M]
    return x

@dataclass
class BeatTriplet:
    '''One annotated beat surrounded by its neighbors.'''
    signal: np.ndarray   # raw ECG excerpt
    cp_left:  int        # LOCAL index of left neighbor cp (cp_{j-1})
    cp_main:  int        # LOCAL index of main beat cp (cp_j)
    cp_right: int        # LOCAL index of right neighbor cp (cp_{j+1})


def extract_main_beat(triplet: BeatTriplet) -> Tuple[np.ndarray, int, int]:
    '''Extract main beat segment with boundaries:
        bl = cp_main - 0.5 * d1
        br = cp_main + 0.6 * d2
    Returns: (segment, bl, br) -- all indices are local to triplet.signal
    '''
    d1 = triplet.cp_main - triplet.cp_left
    d2 = triplet.cp_right - triplet.cp_main

    # BUG 1 FIXED: cp_* are ALREADY local indices. No need to subtract offset.
    bl = triplet.cp_main - int(0.5 * d1)
    br = triplet.cp_main + int(0.6 * d2)

    # Safety bounds to prevent out-of-index slicing
    bl = max(0, bl)
    br = min(len(triplet.signal), br)

    return triplet.signal[bl:br], bl, br


class PositiveAugmenter:
    '''Creates 12 positive (valid heartbeat) samples per beat.
    Fig.4: 1 main + 6 shift (+/-4%, +/-8%, +/-12% of s) + 5 trim
    '''

    SHIFT_FRACS = (0.04, 0.08, 0.12)   # applied left AND right
    TRIM_FRACS  = (0.08, 0.16, 0.20, 0.24, 0.30)  # trimmed from RIGHT of window

    def __call__(self, triplet: BeatTriplet, M: int = 512) -> List[np.ndarray]:
        '''Returns list of 12 raw segments (before align+normalize).'''
        beat, bl, br = extract_main_beat(triplet)
        s = len(beat)
        samples = []

        # 1) Main beat (centred)
        samples.append(triplet.signal[bl:br])

        # 2) Shifted versions (left and right)
        for frac in self.SHIFT_FRACS:
            shift = int(frac * s)
            # Left shift: window moves left -> beat appears right-of-centre
            if bl - shift >= 0 and br - shift <= len(triplet.signal):
                samples.append(triplet.signal[bl - shift : br - shift])
            # Right shift: window moves right -> beat appears left-of-centre
            if bl + shift >= 0 and br + shift <= len(triplet.signal):
                samples.append(triplet.signal[bl + shift : br + shift])

        # 3) Trimmed versions
        # BUG 4 FIXED: Trim from the RIGHT end (decrease br) as per paper Fig.4
        for frac in self.TRIM_FRACS:
            trim = int(frac * s)
            new_br = br - trim
            if new_br > bl:
                samples.append(triplet.signal[bl:new_br])

        # Guarantee exactly 12 (pad with main if augmentations fail at edges)
        while len(samples) < 12:
            samples.append(triplet.signal[bl:br])

        return samples[:12]


class NegativeAugmenter:
    '''Creates 12 negative (non-valid) samples per beat.
    '''
    def __call__(self, triplet: BeatTriplet) -> List[np.ndarray]:
        '''Returns list of 12 negative raw segments.'''
        beat, bl, br = extract_main_beat(triplet)
        s = len(beat)
        samples = []

        # Type 1: Extreme left shift (beats near leftmost corner)
        for frac in (0.35, 0.45, 0.55):
            shift = int(frac * s)
            new_bl = max(0, bl - shift)
            new_br = new_bl + s
            if new_br <= len(triplet.signal):
                samples.append(triplet.signal[new_bl:new_br])

        # Type 2: Extreme right shift (beats near rightmost corner)
        for frac in (0.35, 0.45, 0.55):
            shift = int(frac * s)
            new_br = min(len(triplet.signal), br + shift)
            new_bl = new_br - s
            if new_bl >= 0:
                samples.append(triplet.signal[new_bl:new_br])

        # Type 3: Two-cp segments (spans two beats -- clearly negative)
        two_cp_span = triplet.signal[triplet.cp_left : triplet.cp_right]
        if len(two_cp_span) > 0 and len(two_cp_span) <= 512:
            samples.append(two_cp_span)

        # Type 4: Heavy trim (almost no QRS visible - trim from left this time to ruin morphology)
        for frac in (0.70, 0.80):
            trim = int(frac * s)
            new_bl = bl + trim
            if new_bl < br:
                samples.append(triplet.signal[new_bl:br])

        # Guarantee exactly 12 samples
        while len(samples) < 12:
            samples.append(triplet.signal[bl : bl + max(1, s // 2)])

        return samples[:12]
