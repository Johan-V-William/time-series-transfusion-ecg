import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from config.config import CFG
from src.datasets.seg_real_dts import build_kfold_loaders
from src.models.seg_beat_cls_cnn import build_cnn
from src.train.seg_trainer import Trainer


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


def setup_run_dirs() -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(CFG.training.log_dir) / "seg_beat_cls_smoke" / timestamp
    dirs = {
        "run_root": run_root,
        "checkpoints": run_root / "checkpoints",
        "metrics": run_root / "metrics",
        "figures": run_root / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_file: Path):
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def save_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    run_dirs = setup_run_dirs()
    setup_logging(run_dirs["run_root"] / "train.log")

    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    logger.info("Run root: %s", run_dirs["run_root"])

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

    save_json(
        run_dirs["run_root"] / "run_config.json",
        {
            "mode": "smoke_test",
            "device": device,
            "data_dir": str(data_dir),
            "records": records,
            "n_folds": n_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "checkpoint_dir": str(run_dirs["checkpoints"]),
            "metrics_dir": str(run_dirs["metrics"]),
        },
    )

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
            checkpoint_dir=run_dirs["checkpoints"],
            metrics_dir=run_dirs["metrics"],
        )

        trainer.fit(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            fold_idx=fold_idx,
        )

        metrics = trainer.evaluate(test_loader)
        metrics["fold_idx"] = fold_idx
        all_test_metrics.append(metrics)
        logger.info("Fold %d final -> %s", fold_idx, metrics)

        save_json(run_dirs["metrics"] / f"fold_{fold_idx}_final_metrics.json", metrics)

    aggregate = {}
    for key in ["acc", "Se", "P+", "Sp", "F1"]:
        vals = [m.get(key, 0) for m in all_test_metrics]
        aggregate[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        logger.info("Overall %s: %.4f ± %.4f", key, aggregate[key]["mean"], aggregate[key]["std"])

    save_json(
        run_dirs["run_root"] / "run_summary.json",
        {
            "fold_metrics": all_test_metrics,
            "aggregate_metrics": aggregate,
        },
    )


if __name__ == "__main__":
    main()
