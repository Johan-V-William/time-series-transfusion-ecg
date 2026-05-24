"""
Usage
─────
    # dùng config mặc định
    python train.py

    # override bất kỳ tham số nào qua CLI (Hydra-style dot notation)
    python train.py training.epochs=2000 diffusion.beta_schedule=linear

    # chạy với config khác
    python train.py --config-dir config --config-name train_config

Thứ tự ưu tiên: CLI args > train_config.yaml > default
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
from omegaconf import OmegaConf

from src.datasets.gen_real_dts_v0 import build_datasets
from src.train.gen_trainer import ModelTrainer
from src.evaluation.gen_evaluator import GenerationEvaluator
from src.models.generator_transfusion import GaussianDiffusion1D, TransEncoder


# ──────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────

def load_configs(data_cfg_path: str, train_cfg_path: str):
    data_cfg  = OmegaConf.load(data_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)
    return data_cfg, train_cfg


# ──────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────

def build_model(train_cfg, data_cfg, features: int, device: str):
    """
    Xây dựng TransEncoder + GaussianDiffusion1D từ config.

    seq_len được lấy từ windowing config vì ECG tự tính.
    """
    mcfg  = train_cfg.model
    dcfg  = train_cfg.diffusion
    ecg   = data_cfg.ecg

    # seq_len phụ thuộc vào windowing strategy
    w = ecg.windowing
    if w.method == "hard_fixed":
        seq_len = w.seq_len
    elif w.method == "rpeak":
        fs            = ecg.bandpass_filter.fs
        pre_samples   = round(w.pre_peak_ms  * 1e-3 * fs)
        post_samples  = round(w.post_peak_ms * 1e-3 * fs)
        seq_len       = pre_samples + post_samples
    else:   # adaptive — target_len phải được set
        seq_len = w.target_len

    model = TransEncoder(
        features   = features,
        latent_dim = mcfg.hidden_dim,
        num_heads  = mcfg.n_heads,
        num_layers = mcfg.num_layers,
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length    = seq_len,
        timesteps     = dcfg.timesteps,
        objective     = dcfg.objective,
        loss_type     = dcfg.loss_type,
        beta_schedule = dcfg.beta_schedule,
    )

    return diffusion.to(device)


# ──────────────────────────────────────────────────────────────
# DataLoader builder
# ──────────────────────────────────────────────────────────────

def build_loaders(train_cfg, data_cfg):
    """
    Trả về (train_loader, test_loader, features).

    ECG data shape: (N, seq_len, leads) → transpose → (N, leads, seq_len)
    vì DDPM expects (N, channels, length).
    """
    train_ds, test_ds = build_datasets(data_cfg)

    def collate(batch):
        import torch, numpy as np
        arr = torch.stack(batch)           # (B, seq_len, leads)
        return arr.permute(0, 2, 1)        # (B, leads, seq_len)

    batch_size = train_cfg.training.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=len(test_ds),
        shuffle=False,
        collate_fn=collate,
    )

    # features = leads — bất kỳ sample nào cũng cho biết
    sample = next(iter(train_loader))
    features = sample.shape[1]

    return train_loader, test_loader, features


# ──────────────────────────────────────────────────────────────
# Run name builder
# ──────────────────────────────────────────────────────────────

def make_run_name(train_cfg, data_cfg) -> str:
    d  = train_cfg.diffusion
    w  = data_cfg.ecg.windowing
    return (
        f"ecg-{w.method}"
        f"-{d.beta_schedule}"
        f"-{d.objective}"
        f"-t{d.timesteps}"
    )


def maybe_postprocess_generated_labels(train_cfg, trainer):
    post_cfg = getattr(train_cfg, "postprocess_generated", None)
    if not post_cfg or not post_cfg.get("enabled", False):
        return

    synth_files = sorted(trainer.folder.glob("synth-epoch*.npy"))
    if not synth_files:
        print("[postprocess] No generated .npy file found to label.")
        return

    input_path = synth_files[-1]
    output_name = post_cfg.get("output_name", f"{input_path.stem}-labeled.pt")
    output_path = trainer.folder / output_name

    cmd = [
        sys.executable,
        "scripts/postprocess_generated_labels.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--dataset-type",
        str(post_cfg.get("dataset_type", "mit")),
    ]

    annotation_file = post_cfg.get("annotation_file", None)
    use_model_for_label = post_cfg.get("use_model_for_label", False)
    checkpoint = post_cfg.get("checkpoint", None)

    if annotation_file:
        cmd.extend(["--annotation-file", str(annotation_file)])
    elif use_model_for_label:
        cmd.append("--use-model-for-label")
        if checkpoint:
            cmd.extend(["--checkpoint", str(checkpoint)])

    print("[postprocess] Running external labeling pipeline...")
    subprocess.run(cmd, check=True)
    print(f"[postprocess] Labeled dataset saved to: {output_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECG DDPM Training")
    parser.add_argument("--data-config",  default="config/data.yaml")
    parser.add_argument("--train-config", default="config/train.yaml")
    parser.add_argument("--resume",       default=None,
                        help="Path to checkpoint .pth to resume from")
    # override bất kỳ tham số qua CLI: key=value
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides, e.g. training.epochs=1000")
    args = parser.parse_args()

    # ── load + merge config ─────────────────────────────────
    data_cfg, train_cfg = load_configs(args.data_config, args.train_config)

    # apply CLI overrides  (e.g. "training.epochs=1000")
    for override in args.overrides:
        key, val = override.split("=", 1)
        OmegaConf.update(train_cfg, key, OmegaConf.create(val), merge=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── data ────────────────────────────────────────────────
    train_loader, test_loader, features = build_loaders(train_cfg, data_cfg)
    print(f"Features (leads): {features}")

    # ── model ───────────────────────────────────────────────
    diffusion = build_model(train_cfg, data_cfg, features, device)

    # ── evaluator ───────────────────────────────────────────
    evaluator = GenerationEvaluator(train_cfg)

    # ── trainer ─────────────────────────────────────────────
    run_name = make_run_name(train_cfg, data_cfg)
    trainer  = ModelTrainer(
        cfg          = train_cfg,
        diffusion    = diffusion,
        train_loader = train_loader,
        test_loader  = test_loader,
        device       = device,
        run_name     = run_name,
        evaluator    = evaluator,
    )

    # persist full merged config
    trainer.save_params(OmegaConf.to_container(train_cfg, resolve=True))

    # ── resume nếu có ───────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # ── train ───────────────────────────────────────────────
    trainer.train(start_epoch=start_epoch)

    # ── optional external postprocess for generated labels ──
    maybe_postprocess_generated_labels(train_cfg, trainer)


if __name__ == "__main__":
    main()
