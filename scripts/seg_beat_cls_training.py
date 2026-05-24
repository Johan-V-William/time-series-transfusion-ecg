import argparse
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


def setup_run_dirs(run_mode: str) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(CFG.training.log_dir) / f"seg_beat_cls_{run_mode}" / timestamp
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


def build_optimizer(model: torch.nn.Module):
    optimizer_name = CFG.training.optimizer.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=CFG.training.lr,
            weight_decay=CFG.training.weight_decay,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=CFG.training.lr,
            weight_decay=CFG.training.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"Unsupported optimizer: {CFG.training.optimizer}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train beat classification CNN")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a lightweight smoke test (2 records, 2 folds, 1 epoch, batch_size=16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_mode = "smoke" if args.smoke_test else "full"
    run_dirs = setup_run_dirs(run_mode)
    setup_logging(run_dirs["run_root"] / "train.log")

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    logger.info("Run root: %s", run_dirs["run_root"])
    logger.info("Run mode: %s", run_mode)

    data_dir = Path(CFG.training.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError("data directory not found")

    all_records = get_all_records(data_dir)
    if len(all_records) < 2:
        raise ValueError("Need at least 2 records for k-fold training")

    if args.smoke_test:
        records = all_records[:2]
        n_folds = 2
        epochs = 1
        batch_size = 16
    else:
        num_records = CFG.training.num_records_for_test
        records = all_records[:num_records] if num_records else all_records
        n_folds = CFG.training.n_folds
        epochs = CFG.training.epochs
        batch_size = CFG.training.batch_size

    if len(records) < 2:
        raise ValueError("Need at least 2 records after record selection")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_folds > len(records):
        raise ValueError(f"n_folds={n_folds} cannot exceed number of records={len(records)}")

    logger.info("Selected %d records", len(records))
    logger.info("Records: %s", records)
    logger.info(
        "Training setup -> n_folds=%d | epochs=%d | batch_size=%d | optimizer=%s | lr=%s | weight_decay=%s",
        n_folds,
        epochs,
        batch_size,
        CFG.training.optimizer,
        CFG.training.lr,
        CFG.training.weight_decay,
    )

    save_json(
        run_dirs["run_root"] / "run_config.json",
        {
            "mode": run_mode,
            "device": device,
            "seed": args.seed,
            "data_dir": str(data_dir),
            "records": records,
            "n_folds": n_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": CFG.training.optimizer,
            "lr": CFG.training.lr,
            "weight_decay": CFG.training.weight_decay,
            "checkpoint_dir": str(run_dirs["checkpoints"]),
            "metrics_dir": str(run_dirs["metrics"]),
            "cnn": {
                "conv_channels": list(CFG.cnn.conv_channels),
                "kernel_sizes": list(CFG.cnn.kernel_sizes),
                "pool_size": CFG.cnn.pool_size,
                "fc_hidden": CFG.cnn.fc_hidden,
                "n_classes": CFG.cnn.n_classes,
                "dropout": CFG.cnn.dropout,
                "input_length": CFG.cnn.input_length,
            },
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

        optimizer = build_optimizer(model)

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
