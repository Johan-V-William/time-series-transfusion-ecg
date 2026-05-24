"""
==========
Public API
──────────
    trainer = ModelTrainer(cfg, diffusion, train_loader, test_loader, device)
    trainer.train()
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ModelTrainer:
    """
    Training loop cho GaussianDiffusion1D.

    Parameters
    ----------
    cfg          : DictConfig  — train_config.yaml
    diffusion    : nn.Module   — GaussianDiffusion1D (đã .to(device))
    train_loader : DataLoader
    test_loader  : DataLoader  — dùng để lấy real_data và sample
    device       : str
    run_name     : str         — tên run (dùng cho folder + tensorboard)
    evaluator    : GenerationEvaluator | None
                   nếu None thì bỏ qua evaluation định kỳ
    """

    def __init__(
        self,
        cfg: DictConfig,
        diffusion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: str,
        run_name: str,
        evaluator=None,
    ):
        self.cfg = cfg
        self.tcfg = cfg.training
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.evaluator = evaluator

        self.folder = pathlib.Path(self.tcfg.save_dir) / f"{time.time():.4f}-{run_name}"
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "output").mkdir(exist_ok=True)

        self.optimizer = torch.optim.Adam(
            diffusion.parameters(),
            lr=self.tcfg.lr,
            betas=tuple(self.tcfg.betas),
        )

        self.writer = SummaryWriter(
            log_dir=str(self.folder),
            comment=run_name,
            flush_secs=45,
        )

        self.real_batch = next(iter(test_loader))
        self.real_data, self.real_labels = self._split_batch(self.real_batch)

    def _split_batch(self, batch):
        """
        Hỗ trợ nhiều kiểu batch:
        - tensor                        -> (samples, None)
        - (samples, labels) / [..]      -> (samples, labels)
        - dict{"samples": ..., ...}    -> (samples, labels)
        """
        if isinstance(batch, torch.Tensor):
            return batch, None

        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2:
                samples, labels = batch[0], batch[1]
                return samples, labels
            if len(batch) == 1:
                return batch[0], None

        if isinstance(batch, dict):
            samples = batch.get("samples")
            labels = batch.get("labels")
            if samples is None:
                raise ValueError("Batch dict must contain 'samples'")
            return samples, labels

        raise TypeError(f"Unsupported batch type for evaluator cache: {type(batch)!r}")

    def save_checkpoint(self, epoch: int, tag: str = "final"):
        path = self.folder / f"checkpoint-{tag}.pth"
        torch.save(
            {
                "epoch": epoch,
                "diffusion_state_dict": self.diffusion.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.diffusion.load_state_dict(ckpt["diffusion_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"]

    def save_params(self, params: dict):
        with open(self.folder / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def _sample(self, n: int) -> np.ndarray:
        """
        Sinh n samples từ diffusion.
        Returns (n, seq_len, features) — đã transpose về (N, L, F).
        """
        with torch.no_grad():
            samples = self.diffusion.sample(n)
        samples = samples.cpu().numpy().transpose(0, 2, 1)
        return samples


    def _on_log(self, epoch: int, loss: float):
        print(f"Epoch {epoch + 1:>5}  loss={loss:.6f}")
        self.writer.add_scalar("Loss/train", loss, epoch)

    def _on_visualise(self, epoch: int):
        n = len(self.real_data)
        samples = self._sample(n)
        real_np = self.real_data.cpu().numpy().transpose(0, 2, 1)
        real_labels = None
        if self.real_labels is not None:
            real_labels = (
                self.real_labels.detach().cpu().numpy()
                if torch.is_tensor(self.real_labels)
                else np.asarray(self.real_labels)
            )
        npy_path = self.folder / f"synth-epoch{epoch}.npy"
        np.save(npy_path, samples)

        if self.evaluator is not None:
            eval_dir = self.folder / "output" / f"epoch_{epoch}"
            metrics = self.evaluator.evaluate(
                real_np,
                samples,
                eval_dir,
                labels_real=real_labels,
                labels_fake=None,
            )
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"Eval/{k}", v, epoch)

    def train(self, start_epoch: int = 0):
        epochs = self.tcfg.epochs
        log_every = self.tcfg.log_every
        vis_every = self.tcfg.vis_every

        for epoch in tqdm(range(start_epoch, epochs), desc="Training"):
            epoch_loss = self._train_one_epoch()

            if (epoch + 1) % log_every == 0:
                self._on_log(epoch, epoch_loss)

            if epoch % vis_every == 0:
                self._on_visualise(epoch)

        self.save_checkpoint(epochs, tag="final")
        self.writer.close()
        print(f"\nTraining complete. Files saved to: {self.folder}")

    def _train_one_epoch(self) -> float:
        """Chạy qua toàn bộ train_loader, trả về loss cuối batch."""
        self.diffusion.train()
        last_loss = 0.0

        for batch in self.train_loader:
            batch_x, _ = self._split_batch(batch)
            batch_x = batch_x.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion(batch_x)
            loss.backward()
            self.optimizer.step()
            last_loss = loss.item()

        return last_loss
