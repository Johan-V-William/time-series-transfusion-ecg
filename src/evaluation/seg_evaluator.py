import bisect
from typing import Dict, List

import torch


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Standard binary classification metrics for VB=1, NVB=0."""
    preds = logits.argmax(dim=1)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    se = tp / (tp + fn + 1e-8)
    sp = tn / (tn + fp + 1e-8)
    pp = tp / (tp + fp + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {"Se": se, "Sp": sp, "P+": pp, "F1": f1, "acc": acc}


def evaluate_segmentation(
    detected_cps: List[int],
    annotated_cps: List[int],
    margins_ms: List[int] = [25, 50, 75],
    fs: int = 360,
) -> Dict[str, Dict[str, float]]:
    """Evaluate TP/FP/Se/P+ at different matching margins (ms)."""
    results: Dict[str, Dict[str, float]] = {}

    for margin_ms in margins_ms:
        margin_samples = int(margin_ms * fs / 1000)
        matched_ann = set()
        tp = 0
        fp = 0

        for det_cp in detected_cps:
            if not annotated_cps:
                fp += 1
                continue

            pos = bisect.bisect_left(annotated_cps, det_cp)
            candidates = []
            if pos < len(annotated_cps):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)

            if not candidates:
                fp += 1
                continue

            best_idx = min(candidates, key=lambda i: abs(annotated_cps[i] - det_cp))
            min_dist = abs(annotated_cps[best_idx] - det_cp)

            if min_dist <= margin_samples and best_idx not in matched_ann:
                tp += 1
                matched_ann.add(best_idx)
            else:
                fp += 1

        n_ann = len(annotated_cps)
        results[margin_ms] = {
            "TP": tp,
            "FP": fp,
            "Se": tp / n_ann if n_ann > 0 else 0.0,
            "P+": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        }

    return results
