"""
segment_with_ppld.py
====================
Phiên bản segment.py mở rộng – tích hợp PPLDPreprocessor.

Thay đổi so với segment.py gốc:
  1. Import PPLDPreprocessor, PPLDConfig từ seg_preprocess
  2. Sau khi segment() → gọi ppld.preprocess_all() → tensors
  3. Tensors sẵn sàng truyền vào generative model
  4. Postprocess ngược để khôi phục cấu trúc sinh lý

Chạy:
  python segment_with_ppld.py --record data/100 --fold 0
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wfdb

sys.path.insert(0, os.getcwd())

from config.config import CFG
from src.models.seg_beat_cls_cnn import build_cnn
from pipline.adaptive_windowing import AdaptiveWindowSegmenter
from src.preprocessing.seg_preprocess import PPLDConfig, PPLDPreprocessor
from src.ultis.ultis import plot_segmentation

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("segment")


# ─────────────────────────────────────────────────────────────
# Helpers (không đổi so với gốc)
# ─────────────────────────────────────────────────────────────

def load_best_model(checkpoint_dir: Path, fold_idx: int = 0,
                    device: str = "cpu") -> torch.nn.Module:
    model     = build_cnn(CFG.cnn)
    ckpt_path = checkpoint_dir / f"fold_{fold_idx}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    logger.info(f"Loaded model from {ckpt_path} onto {device}")
    return model


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_segmentation(record_path: str, fold_idx: int = 0,
                     method: str = "both", device: str = "cpu"):

    logger.info(f"Starting segmentation for record: {record_path}")
    record_name = Path(record_path).stem

    # ── 1. Load signal ───────────────────────────────────────
    rec    = wfdb.rdrecord(record_path)
    signal = rec.p_signal[:, 0].astype(np.float32)
    logger.info(f"Loaded signal {record_name} with {len(signal)} samples.")

    # ── 2. Load CNN + segment ────────────────────────────────
    model     = load_best_model(CFG.training.checkpoint_dir, fold_idx, device)
    segmenter = AdaptiveWindowSegmenter(
        cnn_model  = model,
        window_cfg = CFG.segmentation,
        signal_cfg = CFG.signal,
        device     = device,
    )
    beats = segmenter.segment(signal)
    logger.info(f"Detected {len(beats)} beats for record {record_name}")

    for i, beat in enumerate(beats[:10]):
        logger.info(
            f"  Beat {i:4d} | cp={beat.cp_abs:6d} "
            f"| I=[{beat.bl_I}, {beat.br_I}] "
            f"| II=[{beat.bl_II}, {beat.br_II}]"
        )

    # ── 3. PPLD Preprocessing ────────────────────────────────
    ppld_cfg = PPLDConfig(
        fs       = CFG.signal.fs,
        L_std_I  = CFG.ppld.L_std_I  if hasattr(CFG, "ppld") else 200,
        L_std_II = CFG.ppld.L_std_II if hasattr(CFG, "ppld") else 4096,
        r_ratio  = 0.4,
        formula  = "fridericia",
    )
    ppld = PPLDPreprocessor(ppld_cfg)

    if method in ("I", "both"):
        tensor_I, metas_I = ppld.preprocess_beats_I(signal, beats)
        logger.info(f"[PPLD-I]  tensor shape: {tensor_I.shape}")
        # → truyền tensor_I vào generative model ...
        # gen_I = generative_model_I(tensor_I)
        # results_I = ppld.postprocess_beats_I(gen_I, beats, metas_I)

    if method in ("II", "both"):
        tensor_II, metas_II = ppld.preprocess_beats_II(signal, beats)
        logger.info(f"[PPLD-II] tensor shape: {tensor_II.shape}")
        # → truyền tensor_II vào generative model ...
        # gen_II = generative_model_II(tensor_II)
        # results_II = ppld.postprocess_beats_II(gen_II, beats, metas_II)

    # ── 4. Visualise ─────────────────────────────────────────
    plot_segmentation(signal, beats,
                      title=f"ECG Segmentation – {record_name}",
                      fs=CFG.signal.fs)

    return beats, (tensor_I if method in ("I","both") else None,
                   tensor_II if method in ("II","both") else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--fold",   type=int, default=0)
    parser.add_argument("--method", choices=["I", "II", "both"], default="both")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_segmentation(args.record, args.fold, args.method, args.device)
