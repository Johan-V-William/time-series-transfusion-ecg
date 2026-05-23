import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader

from src.models.seg_beat_cls_cnn import BeatCNN
from src.evaluation.seg_evaluator import compute_metrics


class Trainer:
    """Handles one fold of training."""

    def __init__(
        self,
        model: BeatCNN,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        checkpoint_dir: Path = Path("checkpoints/"),
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.ckpt_dir = checkpoint_dir
        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        return {"loss": total_loss / max(1, n), "acc": correct / max(1, n)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels = [], []

        for x, y in loader:
            x = x.to(self.device)
            all_logits.append(self.model(x).cpu())
            all_labels.append(y)

        if not all_logits:
            return {"loss": 0.0, "Se": 0.0, "Sp": 0.0, "P+": 0.0, "F1": 0.0, "acc": 0.0}

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        loss = self.criterion(logits, labels).item()
        metrics = compute_metrics(logits, labels)
        metrics["loss"] = loss
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 15,
        fold_idx: int = 0,
    ) -> BeatCNN:
        best_acc = -1.0
        best_state = None

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            test_metrics = self.evaluate(test_loader)

            self.logger.info(
                "Fold %d | Epoch %d/%d | train_loss=%.4f train_acc=%.4f | test_acc=%.4f F1=%.4f",
                fold_idx,
                epoch + 1,
                epochs,
                train_metrics["loss"],
                train_metrics["acc"],
                test_metrics["acc"],
                test_metrics.get("F1", 0.0),
            )

            if test_metrics["acc"] > best_acc:
                best_acc = test_metrics["acc"]
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if self.ckpt_dir and best_state:
            ckpt_path = self.ckpt_dir / f"fold_{fold_idx}_best.pt"
            torch.save(best_state, ckpt_path)
            self.logger.info("Saved fold %d best model -> %s", fold_idx, ckpt_path)

        if best_state:
            self.model.load_state_dict(best_state)
        return self.model
