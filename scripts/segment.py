import argparse
import numpy as np
import torch
import wfdb
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())

from config.config import CFG
from src.models.seg_beat_cls_cnn import build_cnn
from pipline.adaptive_windowing import AdaptiveWindowSegmenter
from src.ultis.ultis import plot_segmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("segment")


def load_best_model(checkpoint_dir: Path, fold_idx: int = 0, device: str = "cpu") -> torch.nn.Module:
    model = build_cnn(CFG.cnn)
    root = Path(CFG.training.log_dir)

    ckpts = sorted(
          root.glob("seg_beat_cls_*/*/checkpoints/fold_0_best.pt"),
          key=lambda p: p.stat().st_mtime
    )

    ckpt_path = ckpts[-1]
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    logger.info(f"Loaded model from {ckpt_path} onto {device}")
    return model


def run_segmentation(record_path: str, fold_idx: int = 0, method: str = "both", device: str = "cpu"):
    logger.info(f"Starting segmentation for record: {record_path}")

    record_name = Path(record_path).stem
    rec = wfdb.rdrecord(record_path)
    signal = rec.p_signal[:, 0].astype(np.float32)
    logger.info(f"Loaded signal {record_name} with {len(signal)} samples.")

    model = load_best_model(CFG.training.checkpoint_dir, fold_idx, device=device)

    segmenter = AdaptiveWindowSegmenter(
        cnn_model=model,
        window_cfg=CFG.segmentation,
        signal_cfg=CFG.signal,
        device=device,
    )

    beats = segmenter.segment(signal)

    logger.info(f"Detected {len(beats)} beats for record {record_name}")
    for i, beat in enumerate(beats[:10]):
        logger.info(
            f"  Beat {i:4d} | cp={beat.cp_abs:6d} "
            f"| I=[{beat.bl_I}, {beat.br_I}] "
            f"| II=[{beat.bl_II}, {beat.br_II}]"
        )

    plot_segmentation(signal, beats, title=f"ECG Segmentation for Record {record_name}", fs=CFG.signal.fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ECG Segmentation.")
    parser.add_argument("--record", type=str, required=True, help="Path to record (e.g., data/100)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for loading trained model checkpoint")
    parser.add_argument("--method", choices=["I", "II", "both"], default="both")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )
    args = parser.parse_args()


    run_segmentation(args.record, args.fold, args.method, args.device)
