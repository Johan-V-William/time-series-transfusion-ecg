import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader

from models.cnn import BeatCNN
from .evaluator import compute_metrics # Import compute_metrics from evaluator


class Trainer:
    \"\"\"\n    Handles one fold of training.
    Called 10x by the k-fold loop in scripts/train.py.
    \"\"\"

    def __init__(
        self,
        model:      BeatCNN,
        optimizer:  torch.optim.Optimizer,
        device:     str = "cpu",
        checkpoint_dir: Path = Path("checkpoints/"),
    ):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()   # handles Softmax internally
        self.device    = device
        self.ckpt_dir  = checkpoint_dir
        if self.ckpt_dir: # Only create if checkpoint_dir is not None
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger    = logging.getLogger(self.__class__.__name__)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        \"\"\"One training epoch. Returns {'loss': ..., 'acc': ...}.\"\"\"
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)          # [B, 2]
            loss   = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += len(y)

        return {"loss": total_loss / n, "acc": correct / n}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        \"\"\"Returns loss, acc, Se, P+, Sp, F1.\"\"\"
        self.model.eval()
        all_logits, all_labels = [], []

        for x, y in loader:
            x = x.to(self.device)
            all_logits.append(self.model(x).cpu())
            all_labels.append(y)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        loss = self.criterion(logits, labels).item()
        metrics = compute_metrics(logits, labels) # Use the imported compute_metrics
        metrics["loss"] = loss
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        test_loader:  DataLoader,
        epochs:       int = 15,
        fold_idx:     int = 0,
    ) -> BeatCNN:
        \"\"\"Full training loop for one fold. Returns best model.\"\"\"
        best_acc = 0.0
        best_state = None

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            test_metrics  = self.evaluate(test_loader)

            self.logger.info(
                f"Fold {fold_idx} | Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"test_acc={test_metrics['acc']:.4f} "
                f"F1={test_metrics.get('F1', 0):.4f}"
            )

            if test_metrics["acc"] > best_acc:
                best_acc   = test_metrics["acc"]
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Save best model for this fold
        if self.ckpt_dir and best_state:
            ckpt_path = self.ckpt_dir / f"fold_{fold_idx}_best.pt"
            torch.save(best_state, ckpt_path)
            self.logger.info(f"Saved fold {fold_idx} best model -> {ckpt_path}")

        if best_state:
            self.model.load_state_dict(best_state)
        return self.model
