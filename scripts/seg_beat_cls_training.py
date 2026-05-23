import os
import sys
sys.path.insert(0, os.getcwd())

import random
import logging
import torch
import numpy as np
from pathlib import Path

from config.config import CFG
from src.datasets.seg_real_dts import build_kfold_loaders
from src.models.seg_beat_cls_cnn import build_cnn
from src.train.seg_trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("train")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %s", seed)


def get_all_records(data_dir: Path) -> list[str]:
    """Return available local MIT-BIH record base paths (without extension)."""
    records = sorted({p.stem for p in data_dir.glob("*.hea")})
    return [str(data_dir / rec) for rec in records]


def main():
    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    data_dir = Path("data")
    if not data_dir.exists():
        raise FileNotFoundError("data directory not found")

    all_records = get_all_records(data_dir)
    if len(all_records) < 2:
        raise ValueError("Need at least 2 records for k-fold smoke test")

    records = all_records[:2]  # smoke test only
    logger.info("Smoke test records: %s", records)

    n_folds = 2
    epochs = 1
    batch_size = 16

    all_test_metrics = []

    for fold_idx, train_loader, test_loader in build_kfold_loaders(
        all_records=records,
        n_folds=n_folds,
        batch_size=batch_size,
        num_workers=0,
    ):
        logger.info("==================================================")
        logger.info("Fold %d/%d", fold_idx + 1, n_folds)
        logger.info("==================================================")

        model = build_cnn(CFG.cnn)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=Path("checkpoints"),
        )

        trainer.fit(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            fold_idx=fold_idx,
        )

        metrics = trainer.evaluate(test_loader)
        all_test_metrics.append(metrics)
        logger.info("Fold %d final -> %s", fold_idx, metrics)

    for key in ["acc", "Se", "P+", "Sp", "F1"]:
        vals = [m.get(key, 0) for m in all_test_metrics]
        logger.info("Overall %s: %.4f ± %.4f", key, np.mean(vals), np.std(vals))


if __name__ == "__main__":
    main()
