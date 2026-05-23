"""
evaluator.py
============
GenerationEvaluator — đánh giá chất lượng synthetic ECG.

Metrics
───────
    PCA                      — visualization
    t-SNE                    — visualization
    MMD                      — distribution distance  (via tsgm)
    Discriminative Score     — real vs fake classifier (via tsgm)
    Predictive Score         — forecast quality        (via tsgm)
    Correlation Similarity   — feature correlation matrix diff
    Autocorrelation Sim.     — temporal structure diff

Public API
──────────
    ev = GenerationEvaluator(cfg)
    metrics = ev.evaluate(real_data, synthetic_data, output_dir)
    # metrics: dict[str, float]  — cũng được lưu ra metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from omegaconf import DictConfig

try:
    from tsgm.metrics import DiscriminativeMetric, PredictiveMetric, MMDMetric
    _TSGM_AVAILABLE = True
except ImportError:
    _TSGM_AVAILABLE = False


class GenerationEvaluator:
    """
    Đánh giá synthetic time-series so với real data.

    Parameters
    ----------
    cfg : DictConfig — evaluation block từ train_config.yaml
        cfg.evaluation.sample_size
        cfg.evaluation.autocorr_lag
        cfg.evaluation.tsne_perplexity
    """

    def __init__(self, cfg: DictConfig):
        self.ecfg = cfg.evaluation

    # ──────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────

    def evaluate(
        self,
        real_data:      np.ndarray,   # (N, seq_len, features)
        synthetic_data: np.ndarray,   # (N, seq_len, features)
        output_dir:     str | Path,
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ metrics, lưu figures + metrics.json.

        Returns dict[str, float] để caller ghi tensorboard.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        real_data      = np.asarray(real_data)
        synthetic_data = np.asarray(synthetic_data)

        # căn bằng số lượng mẫu
        n = min(len(real_data), len(synthetic_data), self.ecfg.sample_size)
        idx_real = np.random.permutation(len(real_data))[:n]
        idx_fake = np.random.permutation(len(synthetic_data))[:n]
        real = real_data[idx_real]
        fake = synthetic_data[idx_fake]

        metrics: Dict[str, Any] = {}

        # ── distribution metrics ───────────────────────────
        if _TSGM_AVAILABLE:
            metrics["mmd"]                  = self._mmd(real, fake)
            metrics["discriminative_score"] = self._discriminative(real, fake)
            metrics["predictive_score"]     = self._predictive(real, fake)
        else:
            print("[evaluator] tsgm not installed — skipping MMD/Disc/Pred metrics.")

        # ── correlation metrics ────────────────────────────
        metrics["correlation_similarity"]     = self._correlation_sim(real, fake)
        metrics["autocorrelation_similarity"] = self._autocorr_sim(real, fake)

        # ── visualisations ─────────────────────────────────
        self._save_pca(real,  fake, output_dir / "pca.png")
        self._save_tsne(real, fake, output_dir / "tsne.png")
        self._save_sample_traces(real, fake, output_dir / "traces.png")

        # ── persist ───────────────────────────────────────
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    # ──────────────────────────────────────────────────────
    # Visualisations
    # ──────────────────────────────────────────────────────

    def _save_pca(
        self,
        real: np.ndarray,
        fake: np.ndarray,
        save_path: Path,
    ):
        X_real = real.reshape(len(real), -1)
        X_fake = fake.reshape(len(fake), -1)

        pca      = PCA(n_components=2).fit(X_real)
        emb_real = pca.transform(X_real)
        emb_fake = pca.transform(X_fake)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb_real[:, 0], emb_real[:, 1], alpha=0.45, label="Real",      s=12)
        ax.scatter(emb_fake[:, 0], emb_fake[:, 1], alpha=0.45, label="Synthetic", s=12)
        ax.set_title("PCA")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def _save_tsne(
        self,
        real: np.ndarray,
        fake: np.ndarray,
        save_path: Path,
    ):
        X_real = real.reshape(len(real), -1)
        X_fake = fake.reshape(len(fake), -1)
        X      = np.concatenate([X_real, X_fake], axis=0)

        tsne = TSNE(
            n_components=2,
            perplexity=self.ecfg.tsne_perplexity,
            random_state=42,
            verbose=0,
        )
        emb = tsne.fit_transform(X)

        n_real = len(X_real)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb[:n_real, 0], emb[:n_real, 1], alpha=0.45, label="Real",      s=12)
        ax.scatter(emb[n_real:, 0], emb[n_real:, 1], alpha=0.45, label="Synthetic", s=12)
        ax.set_title("t-SNE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def _save_sample_traces(
        self,
        real: np.ndarray,
        fake: np.ndarray,
        save_path: Path,
        n_traces: int = 3,
    ):
        """Vẽ n_traces cặp (real / synthetic) để so sánh trực quan."""
        n_traces = min(n_traces, len(real), len(fake))
        fig, axes = plt.subplots(n_traces, 2, figsize=(12, 3 * n_traces))
        if n_traces == 1:
            axes = axes[np.newaxis, :]

        for i in range(n_traces):
            axes[i, 0].plot(real[i])
            axes[i, 0].set_title(f"Real #{i}")
            axes[i, 1].plot(fake[i])
            axes[i, 1].set_title(f"Synthetic #{i}")

        fig.suptitle("Sample Traces — Real vs Synthetic", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    # ──────────────────────────────────────────────────────
    # Distribution metrics  (tsgm)
    # ──────────────────────────────────────────────────────

    def _mmd(self, real: np.ndarray, fake: np.ndarray) -> float:
        return float(MMDMetric()(real, fake))

    def _discriminative(self, real: np.ndarray, fake: np.ndarray) -> float:
        return float(DiscriminativeMetric()(real, fake))

    def _predictive(self, real: np.ndarray, fake: np.ndarray) -> float:
        return float(PredictiveMetric()(real, fake))

    # ──────────────────────────────────────────────────────
    # Correlation metrics
    # ──────────────────────────────────────────────────────

    def _correlation_sim(self, real: np.ndarray, fake: np.ndarray) -> float:
        """
        Mean absolute difference of feature correlation matrices.
        Lower = more similar.
        """
        flat_real = real.reshape(len(real), -1)
        flat_fake = fake.reshape(len(fake), -1)
        corr_real = np.corrcoef(flat_real.T)
        corr_fake = np.corrcoef(flat_fake.T)
        return float(np.mean(np.abs(corr_real - corr_fake)))

    def _autocorr_sim(self, real: np.ndarray, fake: np.ndarray) -> float:
        """
        Mean absolute difference of average autocorrelation functions.
        Lower = more similar temporal structure.
        """
        lag = self.ecfg.autocorr_lag

        def acf_series(x: np.ndarray) -> np.ndarray:
            """x: (seq_len, features) → mean ACF over features, lags 1..lag-1"""
            x = x.flatten()
            return np.array([pearsonr(x[:-l], x[l:])[0] for l in range(1, lag)])

        real_acf = np.mean([acf_series(s) for s in real], axis=0)
        fake_acf = np.mean([acf_series(s) for s in fake], axis=0)
        return float(np.mean(np.abs(real_acf - fake_acf)))
