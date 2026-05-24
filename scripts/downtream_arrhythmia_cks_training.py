import os
import sys
sys.path.insert(0, os.getcwd())

import random
import logging # Added import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional # Import Optional

from config.config import CFG, CNNConfig
from src.datasets.downstream_prepare_tstr_dts import build_kfold_loaders
from models.cnn import build_cnn
from training.trainer import Trainer

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
    logger.info("Global seed set to {}".format(seed))

def get_all_records(data_dir: Path) -> list:
    """Return sorted list of MIT-BIH record paths (without extension)."""
    # MIT-BIH has 48 records: 100-124, 200-234 (excluding some)
    MITBIH_RECORDS = [
        "100","101","102","103","104","105","106","107","108","109",
        "111","112","113","114","115","116","117","118","119",
        "121","122","123","124",
        "200","201","202","203","205","207","208","209","210",
        "212","213","214","215","217","219","220","221","222","223",
        "228","230","231","232","233","234",
    ]
    return [str(data_dir / rec) for rec in MITBIH_RECORDS]


def main():
    seed_everything(42) # Seed everything for reproducibility

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: {}".format(device))

    all_records = get_all_records(CFG.training.data_dir)

    # Use subset of records if specified in config for quick testing
    num_records_for_test = CFG.training.num_records_for_test
    if num_records_for_test is not None and 0 < num_records_for_test < len(all_records):
        records = all_records[:num_records_for_test]
        logger.info("Loading a subset of {} records for testing/quick run.".format(num_records_for_test))
    else:
        records = all_records
        logger.info("Loading all {} records for training.".format(len(records)))

    all_test_metrics = []

    for fold_idx, train_loader, test_loader in build_kfold_loaders(
        all_records=records,
        n_folds=CFG.training.n_folds,
        batch_size=CFG.training.batch_size,
    ):
        logger.info("==================================================")
        logger.info("Fold {}/{}".format(fold_idx + 1, CFG.training.n_folds))
        logger.info("==================================================")
        cnn_name = getattr(CFG.cnn, "name", "custom_optimized") if isinstance(CFG.cnn, CNNConfig) else "custom_optimized"
        model = build_cnn(CFG.cnn)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CFG.training.lr,
            weight_decay=CFG.training.weight_decay,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=CFG.training.checkpoint_dir,
        )

        best_model = trainer.fit(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=CFG.training.epochs,
            fold_idx=fold_idx,
        )

        # Final evaluation on this fold
        metrics = trainer.evaluate(test_loader)
        all_test_metrics.append(metrics)
        logger.info("Fold {} final -> {}".format(fold_idx, metrics))

    for key in ["acc", "Se", "P+", "Sp", "F1"]:
        vals = [m.get(key, 0) for m in all_test_metrics]
        logger.info("Overall {}: {:.4f} \u00B1 {:.4f}".format(key, np.mean(vals), np.std(vals)))


def test_gpu_training():
    """Test training pipeline on GPU with a small dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Testing on device: {}".format(device))

    # Use a small subset of records for testing
    records = get_all_records(CFG.training.data_dir)[:2]  # Use only 2 records for quick test

    # Build data loaders with smaller batch size
    for fold_idx, train_loader, test_loader in build_kfold_loaders(
        all_records=records,
        n_folds=2,  # Use 2 folds for quick test
        batch_size=2,  # Small batch size
    ):
        logger.info("==================================================")
        logger.info("Test Fold {}/2".format(fold_idx + 1))
        logger.info("==================================================")
        model = build_cnn(CFG.cnn)
        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,  # Small learning rate for testing
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=None,  # No checkpointing for testing
        )

        # Run only 1 epoch for testing
        trainer.fit(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=1,
            fold_idx=fold_idx,
        )

        # Evaluate on test loader
        metrics = trainer.evaluate(test_loader)
        logger.info("Test Fold {} metrics -> {}".format(fold_idx, metrics))

if __name__ == "__main__":
    # You can call main() for full training or test_gpu_training() for a quick test.
    # To test main with a subset, you can temporarily modify CFG.training.num_records_for_test
    # e.g., CFG.training.num_records_for_test = 5
    main()
