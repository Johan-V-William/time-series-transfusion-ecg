from __future__ import annotations

import os
import json
import time
import pathlib
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from omegaconf import OmegaConf

from src.models.generator_transfusion import (
    GaussianDiffusion1D,
    TransEncoder,
)

from src.datasets.gen_real_dts_v0 import build_datasets
from src.evaluation.gen_evaluator import GenerationEvaluator


# ============================================================
# CONFIG
# ============================================================

def load_configs(
    data_cfg_path: str,
    train_cfg_path: str,
):
    data_cfg = OmegaConf.load(data_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)

    return data_cfg, train_cfg


# ============================================================
# DATA
# ============================================================

def build_loaders(
    data_cfg,
    batch_size: int,
):
    train_ds, test_ds = build_datasets(data_cfg)

    def collate(batch):
        x = torch.stack(batch)      # (B,L,F)
        return x.permute(0, 2, 1)   # (B,F,L)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=len(test_ds),
        shuffle=False,
        collate_fn=collate,
    )

    return train_loader, test_loader, train_ds, test_ds


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-config",
        default="config/data.yaml",
    )

    parser.add_argument(
        "--train-config",
        default="config/train.yaml",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configs
    # --------------------------------------------------------

    data_cfg, train_cfg = load_configs(
        args.data_config,
        args.train_config,
    )

    # --------------------------------------------------------
    # Config shortcuts
    # --------------------------------------------------------

    tcfg = train_cfg.training
    mcfg = train_cfg.model
    dcfg = train_cfg.diffusion

    epochs = tcfg.epochs
    batch_size = tcfg.batch_size

    latent_dim = mcfg.hidden_dim
    num_layers = mcfg.num_layers
    n_heads = mcfg.n_heads

    timesteps = dcfg.timesteps
    beta_schedule = dcfg.beta_schedule
    objective = dcfg.objective
    loss_type = dcfg.loss_type

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_loader,
        test_loader,
        train_ds,
        test_ds,
    ) = build_loaders(
        data_cfg=data_cfg,
        batch_size=batch_size,
    )

    sample = next(iter(train_loader))

    features = sample.shape[1]
    seq_len = sample.shape[2]

    print(f"Features : {features}")
    print(f"Seq Len  : {seq_len}")

    real_data = next(iter(test_loader))

    real_np = (
        real_data
        .cpu()
        .numpy()
        .transpose(0, 2, 1)
    )

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    architecture = "custom-transformers"

    file_name = (
        f"{architecture}"
        f"-ecg"
        f"-{loss_type}"
        f"-{beta_schedule}"
        f"-{seq_len}"
        f"-{objective}"
    )

    folder_name = (
        f"saved_files/"
        f"{time.time():.4f}-"
        f"{file_name}"
    )

    pathlib.Path(folder_name).mkdir(
        parents=True,
        exist_ok=True,
    )

    pathlib.Path(
        f"{folder_name}/output"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # save merged config

    with open(
        f"{folder_name}/params.json",
        "w",
    ) as f:

        json.dump(
            OmegaConf.to_container(
                train_cfg,
                resolve=True,
            ),
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Tensorboard
    # --------------------------------------------------------

    writer = SummaryWriter(
        log_dir=folder_name,
        comment=file_name,
        flush_secs=45,
    )

    # --------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------

    evaluator = GenerationEvaluator(
        train_cfg
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TransEncoder(
        features=features,
        latent_dim=latent_dim,
        num_heads=n_heads,
        num_layers=num_layers,
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=seq_len,
        timesteps=timesteps,
        objective=objective,
        loss_type=loss_type,
        beta_schedule=beta_schedule,
    )

    diffusion = diffusion.to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        diffusion.parameters(),
        lr=tcfg.lr,
        betas=tuple(tcfg.betas),
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for epoch in tqdm(
        range(epochs),
        desc="Training",
    ):

        diffusion.train()

        for i, batch in enumerate(train_loader):

            batch = batch.to(device)

            optimizer.zero_grad()

            loss = diffusion(batch)

            loss.backward()

            optimizer.step()

            # ----------------------------------------
            # Tensorboard
            # ----------------------------------------

            if i % len(train_loader) == 0:

                writer.add_scalar(
                    "Loss/train",
                    loss.item(),
                    epoch,
                )

            # ----------------------------------------
            # Logging
            # ----------------------------------------

            if (
                i % len(train_loader) == 0
                and epoch % tcfg.log_every == 0
            ):

                print(
                    f"Epoch {epoch+1} "
                    f"Loss={loss.item():.6f}"
                )

            # ----------------------------------------
            # Evaluation
            # ----------------------------------------

            if (
                i % len(train_loader) == 0
                and epoch % tcfg.vis_every == 0
            ):

                with torch.no_grad():

                    samples = diffusion.sample(
                        len(test_ds)
                    )

                    samples = (
                        samples.cpu()
                        .numpy()
                        .transpose(0, 2, 1)
                    )

                save_dir = os.path.join(
                    folder_name,
                    "output",
                    f"epoch_{epoch}",
                )

                os.makedirs(
                    save_dir,
                    exist_ok=True,
                )

                metrics = evaluator.evaluate(
                    real_np,
                    samples,
                    save_dir,
                )

                for k, v in metrics.items():

                    if isinstance(
                        v,
                        (float, int),
                    ):
                        writer.add_scalar(
                            f"Eval/{k}",
                            v,
                            epoch,
                        )

    # --------------------------------------------------------
    # Save final checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epochs,
            "diffusion_state_dict":
                diffusion.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
        },
        os.path.join(
            folder_name,
            "checkpoint-final.pth",
        ),
    )

    writer.close()

    print()
    print(
        f"Training complete.\n"
        f"Saved to: {folder_name}"
    )


if __name__ == "__main__":
    main()
