import torch
import numpy as np
import bisect
from typing import Dict, List

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    \"\"\"\n    Standard binary classification metrics.
    VB=1 (positive class), NVB=0 (negative class).
    \"\"\"
    preds = logits.argmax(dim=1)

    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    Se  = TP / (TP + FN + 1e-8)          # Sensitivity (Recall)
    Sp  = TN / (TN + FP + 1e-8)          # Specificity
    Pp  = TP / (TP + FP + 1e-8)          # Precision (P+)
    F1  = 2 * TP / (2 * TP + FP + FN + 1e-8)
    Acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return {"Se": Se, "Sp": Sp, "P+": Pp, "F1": F1, "acc": Acc}


def evaluate_segmentation(
    detected_cps:   List[int],
    annotated_cps:  List[int],
    margins_ms:     List[int] = [25, 50, 75],
    fs:             int = 360,
) -> Dict[str, Dict[str, float]]:
    \"\"\"\n    Table 3 evaluation: TP/FP/Se/P+ at different ms margins.
    Optimized to O(N log M) using binary search.
    \"\"\"
    results = {}

    for margin_ms in margins_ms:
        margin_samples = int(margin_ms * fs / 1000)
        matched_ann    = set()
        TP = 0
        FP = 0

        for det_cp in detected_cps:
            if not annotated_cps:
                FP += 1
                continue

            # OPTIMIZATION: Use binary search to find the closest time point
            pos = bisect.bisect_left(annotated_cps, det_cp)

            # The closest point can only be at position pos or pos - 1
            candidates = []
            if pos < len(annotated_cps):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)

            # Choose the point with the smallest distance
            if not candidates:
                FP += 1
                continue

            best_idx = min(candidates, key=lambda i: abs(annotated_cps[i] - det_cp))
            min_dist = abs(annotated_cps[best_idx] - det_cp)

            if min_dist <= margin_samples and best_idx not in matched_ann:
                TP += 1
                matched_ann.add(best_idx)
            else:
                FP += 1

        N_ann = len(annotated_cps)
        results[margin_ms] = {
            "TP":  TP,
            "FP":  FP,
            # TERMINOLOGY NOTE: In R-peak counting, TP / N_ann is actually
            # Sensitivity (Se) / Recall, not traditional Accuracy (Acc).
            # I changed the label "ACC" to "Se" to standardize with medical terminology.
            "Se":  TP / N_ann if N_ann > 0 else 0.0,
            "P+":  TP / (TP + FP) if (TP + FP) > 0 else 0.0,
        }

    return results
