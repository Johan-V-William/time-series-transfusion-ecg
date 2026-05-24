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
    Downstream Effectiveness — train-on-fake/test-on-real accuracy (via tsgm)
    Correlation Similarity   — feature correlation matrix diff
    Autocorrelation Sim.     — temporal structure diff
    Sig-W1 Metric            — path signature Wasserstein-1 distance
    FFT Reconstruction Error — spectral amplitude difference
    Fréchet Time-Series Dist — FID-style distance in feature space
    HRV Metrics              — SDNN, RMSSD (via NeuroKit2 / scipy)
    Morphological Intervals  — PR interval, QRS duration, QT/QTc

Public API
──────────
    ev = GenerationEvaluator(cfg)
    metrics = ev.evaluate(real_data, synthetic_data, output_dir)
    # metrics: dict[str, float]  — cũng được lưu ra metrics.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from omegaconf import DictConfig

try:
    from tsgm.metrics import (
        DiscriminativeMetric,
        PredictiveMetric,
        MMDMetric,
        DownstreamEffectivenessMetric,
    )
    _TSGM_AVAILABLE = True
except ImportError:
    _TSGM_AVAILABLE = False

try:
    from ultis.sigw1_metrics import SigW1Metric, AddTime, Scale
    _SIGNATORY_AVAILABLE = True
except ImportError:
    _SIGNATORY_AVAILABLE = False

try:
    import neurokit2 as nk
    _NEUROKIT_AVAILABLE = True
except ImportError:
    _NEUROKIT_AVAILABLE = False


class GenerationEvaluator:
    """
    Đánh giá synthetic time-series so với real data.

    Parameters
    ----------
    cfg : DictConfig — evaluation block từ train_config.yaml
        cfg.evaluation.sample_size
        cfg.evaluation.autocorr_lag
        cfg.evaluation.tsne_perplexity
        cfg.evaluation.sampling_rate   — Hz (default 500)
        cfg.evaluation.sigw1           — sub-config cho SigW1 (optional)
        cfg.evaluation.downstream      — sub-config cho DownstreamEffectiveness (optional)
            .enabled        : bool  (default False)
            .classifier     : str   "knn" | "svm" | "rf"  (default "knn")
            .n_splits       : int   số fold CV  (default 5)
            .labels_real    : list  nhãn tương ứng real_data  (required)
            .labels_fake    : list  nhãn tương ứng synthetic_data (required)
    """

    def __init__(self, cfg: DictConfig):
        self.ecfg = cfg.evaluation
        self.fs: int = int(getattr(self.ecfg, "sampling_rate", 500))

    # ──────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────

    def evaluate(
        self,
        real_data:      np.ndarray,             # (N, seq_len, features)
        synthetic_data: np.ndarray,             # (N, seq_len, features)
        output_dir:     str | Path,
        labels_real:    Optional[np.ndarray] = None,   # (N,) nhãn cho real
        labels_fake:    Optional[np.ndarray] = None,   # (N,) nhãn cho synthetic
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
        labels_real_sub = labels_real[idx_real] if labels_real is not None else None
        labels_fake_sub = labels_fake[idx_fake] if labels_fake is not None else None

        metrics: Dict[str, Any] = {}

        # ── distribution metrics  (tsgm) ──────────────────
        if _TSGM_AVAILABLE:
            metrics["mmd"]                  = self._mmd(real, fake)
            metrics["discriminative_score"] = self._discriminative(real, fake)
            metrics["predictive_score"]     = self._predictive(real, fake)

            # ── Downstream Effectiveness ───────────────────
            ds_cfg = getattr(self.ecfg, "downstream", None)
            if ds_cfg and ds_cfg.get("enabled", False):
                if labels_real_sub is not None and labels_fake_sub is not None:
                    metrics["downstream_effectiveness"] = self._downstream(
                        real, fake,
                        labels_real_sub, labels_fake_sub,
                        ds_cfg,
                    )
                else:
                    print(
                        "[evaluator] downstream_effectiveness requires labels_real "
                        "and labels_fake — skipping."
                    )
        else:
            print("[evaluator] tsgm not installed — skipping MMD/Disc/Pred/Downstream metrics.")

        # ── correlation metrics ────────────────────────────
        metrics["correlation_similarity"]     = self._correlation_sim(real, fake)
        metrics["autocorrelation_similarity"] = self._autocorr_sim(real, fake)

        # ── Sig-W1 metric (optional) ───────────────────────
        sigw1_cfg = getattr(self.ecfg, "sigw1", None)
        if sigw1_cfg and sigw1_cfg.get("enabled", False):
            if _SIGNATORY_AVAILABLE:
                metrics["sigw1"] = self._sigw1(real, fake, sigw1_cfg)
            else:
                print("[evaluator] signatory not installed — cannot compute SigW1 metric.")

        # ── FFT Reconstruction Error ───────────────────────
        metrics["fft_reconstruction_error"] = self._fft_error(real, fake)

        # ── Fréchet Time-Series Distance ───────────────────
        metrics["frechet_ts_distance"] = self._frechet_ts_distance(real, fake)

        # ── HRV metrics (requires NeuroKit2) ──────────────
        if _NEUROKIT_AVAILABLE:
            hrv_real = self._extract_hrv_stats(real)
            hrv_fake = self._extract_hrv_stats(fake)
            for key in hrv_real:
                metrics[f"hrv_{key}_real"] = hrv_real[key]
                metrics[f"hrv_{key}_fake"] = hrv_fake[key]
                metrics[f"hrv_{key}_diff"] = abs(hrv_real[key] - hrv_fake[key])
        else:
            print("[evaluator] neurokit2 not installed — skipping HRV metrics.")

        # ── Morphological interval deviations ─────────────
        if _NEUROKIT_AVAILABLE:
            morph_real = self._extract_morphology(real)
            morph_fake = self._extract_morphology(fake)
            for key in morph_real:
                metrics[f"morph_{key}_real"] = morph_real[key]
                metrics[f"morph_{key}_fake"] = morph_fake[key]
                metrics[f"morph_{key}_diff"] = abs(morph_real[key] - morph_fake[key])
        else:
            print("[evaluator] neurokit2 not installed — skipping morphological metrics.")

        # ── visualisations ─────────────────────────────────
        self._save_pca(real,  fake, output_dir / "pca.png")
        self._save_tsne(real, fake, output_dir / "tsne.png")
        self._save_sample_traces(real, fake, output_dir / "traces.png")
        self._save_fft_comparison(real, fake, output_dir / "fft_comparison.png")
        if _NEUROKIT_AVAILABLE:
            self._save_hrv_comparison(real, fake, output_dir / "hrv_comparison.png")

        # ── persist ────────────────────────────────────────
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    # ──────────────────────────────────────────────────────
    # Visualisations
    # ──────────────────────────────────────────────────────

    def _save_pca(self, real: np.ndarray, fake: np.ndarray, save_path: Path):
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

    def _save_tsne(self, real: np.ndarray, fake: np.ndarray, save_path: Path):
        X_real = real.reshape(len(real), -1)
        X_fake = fake.reshape(len(fake), -1)
        X      = np.concatenate([X_real, X_fake], axis=0)

        tsne = TSNE(
            n_components=2,
            perplexity=self.ecfg.tsne_perplexity,
            random_state=42,
            verbose=0,
        )
        emb    = tsne.fit_transform(X)
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

    def _save_fft_comparison(
        self,
        real: np.ndarray,
        fake: np.ndarray,
        save_path: Path,
    ):
        """So sánh biên độ phổ tần số trung bình giữa real và synthetic."""
        def mean_fft_amplitude(data: np.ndarray) -> np.ndarray:
            # data: (n, seq_len, features) — lấy kênh đầu tiên
            signals = data[:, :, 0]
            amps = np.abs(fft(signals, axis=1))
            return np.mean(amps[:, : signals.shape[1] // 2], axis=0)

        freq = fftfreq(real.shape[1], d=1.0 / self.fs)[: real.shape[1] // 2]
        amp_real = mean_fft_amplitude(real)
        amp_fake = mean_fft_amplitude(fake)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(freq, amp_real, label="Real",      alpha=0.8)
        ax.plot(freq, amp_fake, label="Synthetic", alpha=0.8, linestyle="--")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Mean Amplitude")
        ax.set_title("FFT Spectrum — Real vs Synthetic")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def _save_hrv_comparison(
        self,
        real: np.ndarray,
        fake: np.ndarray,
        save_path: Path,
    ):
        """Box-plot so sánh SDNN và RMSSD giữa real và synthetic."""
        def collect_hrv(data: np.ndarray) -> Dict[str, List[float]]:
            sdnn_list, rmssd_list = [], []
            for sample in data:
                sig = sample[:, 0]
                try:
                    _, info = nk.ecg_peaks(sig, sampling_rate=self.fs)
                    rri = np.diff(info["ECG_R_Peaks"]) / self.fs * 1000  # ms
                    if len(rri) >= 2:
                        sdnn_list.append(float(np.std(rri, ddof=1)))
                        rmssd_list.append(float(np.sqrt(np.mean(np.diff(rri) ** 2))))
                except Exception:
                    pass
            return {"SDNN": sdnn_list, "RMSSD": rmssd_list}

        hrv_r = collect_hrv(real)
        hrv_f = collect_hrv(fake)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, key in zip(axes, ["SDNN", "RMSSD"]):
            ax.boxplot([hrv_r[key], hrv_f[key]], labels=["Real", "Synthetic"])
            ax.set_title(f"HRV — {key} (ms)")

        fig.suptitle("HRV Distribution — Real vs Synthetic", fontsize=13)
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

    def _downstream(
        self,
        real:        np.ndarray,
        fake:        np.ndarray,
        labels_real: np.ndarray,
        labels_fake: np.ndarray,
        cfg:         DictConfig,
    ) -> float:
        """
        Downstream Effectiveness (Train-on-Synthetic / Test-on-Real).

        Ý tưởng: nếu synthetic data thực sự hữu ích, một classifier huấn luyện
        hoàn toàn trên synthetic sẽ cho accuracy cao khi test trên real data.
        Metric được tính qua cross-validation để ổn định hơn.

        Cách tsgm triển khai
        ─────────────────────
        DownstreamEffectivenessMetric nhận vào:
            - classifier   : sklearn estimator (KNN / SVM / RF, ...)
            - n_splits     : số fold cho StratifiedKFold CV trên real test set

        Nó sẽ:
            1. Fit classifier trên (fake, labels_fake).
            2. Evaluate trên (real, labels_real) với n_splits-fold CV.
            3. Trả về accuracy trung bình → cao hơn là synthetic tốt hơn.

        Parameters
        ----------
        cfg.classifier : "knn" | "svm" | "rf"  (default "knn")
        cfg.n_splits   : int                    (default 5)

        Returns
        -------
        float — mean downstream accuracy trên real test set.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier

        classifier_name = cfg.get("classifier", "knn").lower()
        n_splits        = int(cfg.get("n_splits", 5))

        # Flatten time-series thành vector đặc trưng cho sklearn classifier
        # shape (n, seq_len, features) → (n, seq_len * features)
        real_2d = real.reshape(len(real), -1)
        fake_2d = fake.reshape(len(fake), -1)

        if classifier_name == "svm":
            clf = SVC(kernel="rbf", C=1.0, probability=False)
        elif classifier_name == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # default: knn
            clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

        metric = DownstreamEffectivenessMetric(
            clf,
            n_splits=n_splits,
        )
        score = metric(
            real_2d,   labels_real,   # test set
            fake_2d,   labels_fake,   # training set (synthetic)
        )
        return float(score)

    # ──────────────────────────────────────────────────────
    # Correlation metrics
    # ──────────────────────────────────────────────────────

    def _correlation_sim(self, real: np.ndarray, fake: np.ndarray) -> float:
        """Mean absolute difference of feature correlation matrices. Lower = better."""
        flat_real = real.reshape(len(real), -1)
        flat_fake = fake.reshape(len(fake), -1)
        corr_real = np.corrcoef(flat_real.T)
        corr_fake = np.corrcoef(flat_fake.T)
        return float(np.mean(np.abs(corr_real - corr_fake)))

    def _autocorr_sim(self, real: np.ndarray, fake: np.ndarray) -> float:
        """Mean absolute difference of average ACF. Lower = better temporal structure."""
        lag = self.ecfg.autocorr_lag

        def acf_series(x: np.ndarray) -> np.ndarray:
            x = x.flatten()
            return np.array([pearsonr(x[:-l], x[l:])[0] for l in range(1, lag)])

        real_acf = np.mean([acf_series(s) for s in real], axis=0)
        fake_acf = np.mean([acf_series(s) for s in fake], axis=0)
        return float(np.mean(np.abs(real_acf - fake_acf)))

    # ──────────────────────────────────────────────────────
    # Sig-W1 metric (optional)
    # ──────────────────────────────────────────────────────

    def _sigw1(self, real: np.ndarray, fake: np.ndarray, cfg: DictConfig) -> float:
        """
        Tính Sig-W1: RMSE giữa expected signature của real và fake.
        real, fake: numpy arrays shape (n, seq_len, features)
        """
        import torch

        real_t = torch.from_numpy(real).float()
        fake_t = torch.from_numpy(fake).float()

        augmentations = []
        aug_names = cfg.get("augmentations", [])
        if "add_time" in aug_names:
            augmentations.append(AddTime())
        if "scale" in aug_names:
            scale_val = cfg.get("scale_value", 1.0)
            augmentations.append(Scale(scale_val))

        depth     = cfg.get("depth", 4)
        normalise = cfg.get("normalise", True)

        metric = SigW1Metric(
            real_data=real_t,
            depth=depth,
            augmentations=tuple(augmentations),
            normalise=normalise,
        )
        score = metric(fake_t)
        return float(score.cpu().numpy())

    # ──────────────────────────────────────────────────────
    # FFT Reconstruction Error
    # ──────────────────────────────────────────────────────

    def _fft_error(self, real: np.ndarray, fake: np.ndarray) -> float:
        """
        Mean absolute difference between average FFT amplitude spectra.

        Thực hiện biến đổi Fourier nhanh trên từng đoạn ECG, lấy biên độ phổ
        tần số trung bình của tập real và tập fake rồi tính MAE.

        Lower = better spectral fidelity.
        """
        def mean_amplitude(data: np.ndarray) -> np.ndarray:
            # (n, seq_len, features) → dùng tất cả features, trung bình qua n
            # shape after fft: (n, seq_len//2, features)
            amps = np.abs(fft(data, axis=1))[:, : data.shape[1] // 2, :]
            return np.mean(amps, axis=0)   # (seq_len//2, features)

        amp_real = mean_amplitude(real)
        amp_fake = mean_amplitude(fake)
        return float(np.mean(np.abs(amp_real - amp_fake)))

    # ──────────────────────────────────────────────────────
    # Fréchet Time-Series Distance (FTD)
    # ──────────────────────────────────────────────────────

    def _frechet_ts_distance(self, real: np.ndarray, fake: np.ndarray) -> float:
        """
        FID-style Fréchet distance computed in the flattened feature space.

        FTD = ||μ_r − μ_f||² + Tr(Σ_r + Σ_f − 2(Σ_r Σ_f)^{1/2})

        Lower = better distributional match.

        Note: đây là approximate FTD không qua neural feature extractor.
        Để có FTD chính xác hơn, thay X_* bằng activation từ một encoder
        đã huấn luyện trên ECG.
        """
        from scipy.linalg import sqrtm

        X_real = real.reshape(len(real), -1).astype(np.float64)
        X_fake = fake.reshape(len(fake), -1).astype(np.float64)

        mu_r, mu_f = X_real.mean(axis=0), X_fake.mean(axis=0)
        sigma_r    = np.cov(X_real, rowvar=False)
        sigma_f    = np.cov(X_fake, rowvar=False)

        diff       = mu_r - mu_f
        cov_mean   = sqrtm(sigma_r @ sigma_f)

        # sqrtm có thể trả về complex do floating-point noise nhỏ
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        ftd = float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * cov_mean))
        return ftd

    # ──────────────────────────────────────────────────────
    # HRV Metrics  (requires NeuroKit2)
    # ──────────────────────────────────────────────────────

    def _extract_hrv_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Trích xuất các chỉ số HRV từ tập dữ liệu ECG.

        Sử dụng NeuroKit2 để phát hiện đỉnh R (Pan-Tompkins), sau đó tính:
            - SDNN  : Độ lệch chuẩn các khoảng R-R (ms) — đo HRV tổng thể.
            - RMSSD : Căn bậc hai trung bình bình phương các hiệu R-R liên tiếp (ms)
                      — đo biến thiên ngắn hạn / phó giao cảm.
            - mean_rr : Khoảng R-R trung bình (ms) — tương ứng nhịp tim trung bình.
            - pnn50  : Tỉ lệ % cặp R-R liên tiếp có |ΔRR| > 50 ms.

        Parameters
        ----------
        data : (n, seq_len, features) — kênh đầu tiên được dùng làm ECG lead.

        Returns
        -------
        dict với giá trị trung bình qua toàn bộ n mẫu.
        """
        sdnn_list, rmssd_list, mrr_list, pnn50_list = [], [], [], []

        for sample in data:
            sig = sample[:, 0]
            try:
                _, info = nk.ecg_peaks(sig, sampling_rate=self.fs)
                r_peaks = info["ECG_R_Peaks"]
                if len(r_peaks) < 3:
                    continue
                rri = np.diff(r_peaks) / self.fs * 1000.0  # samples → ms

                sdnn_list.append(float(np.std(rri, ddof=1)))
                rmssd_list.append(float(np.sqrt(np.mean(np.diff(rri) ** 2))))
                mrr_list.append(float(np.mean(rri)))
                pnn50_list.append(float(np.mean(np.abs(np.diff(rri)) > 50.0) * 100.0))
            except Exception:
                pass  # R-peak detection có thể thất bại trên signal nhiễu nhiều

        def _safe_mean(lst: list) -> float:
            return float(np.mean(lst)) if lst else float("nan")

        return {
            "sdnn_ms":   _safe_mean(sdnn_list),
            "rmssd_ms":  _safe_mean(rmssd_list),
            "mean_rr_ms": _safe_mean(mrr_list),
            "pnn50_pct": _safe_mean(pnn50_list),
        }

    # ──────────────────────────────────────────────────────
    # Morphological Interval Deviations  (requires NeuroKit2)
    # ──────────────────────────────────────────────────────

    def _extract_morphology(self, data: np.ndarray) -> Dict[str, float]:
        """
        Đo và so sánh độ dài (ms) của các khoảng sinh lý ECG:
            - PR interval   : Từ đầu sóng P đến đầu phức bộ QRS.
            - QRS duration  : Thời gian phức bộ QRS.
            - QT interval   : Từ đầu QRS đến cuối sóng T.
            - QTc interval  : QT hiệu chỉnh theo nhịp tim (công thức Bazett:
                              QTc = QT / √(RR/1000)).

        Sử dụng nk.ecg_delineate (wavelet method) để xác định vị trí các
        điểm sóng P, Q, R, S, T trên từng beat.

        Parameters
        ----------
        data : (n, seq_len, features)

        Returns
        -------
        dict với giá trị trung bình (ms) qua toàn bộ beat hợp lệ.
        """
        pr_list, qrs_list, qt_list, qtc_list = [], [], [], []

        for sample in data:
            sig = sample[:, 0]
            try:
                _, rpeaks = nk.ecg_peaks(sig, sampling_rate=self.fs)
                _, waves  = nk.ecg_delineate(
                    sig,
                    rpeaks,
                    sampling_rate=self.fs,
                    method="dwt",
                )

                p_onsets   = np.asarray(waves.get("ECG_P_Onsets",  []), dtype=float)
                q_peaks    = np.asarray(waves.get("ECG_Q_Peaks",   []), dtype=float)
                s_offsets  = np.asarray(waves.get("ECG_S_Offsets", []), dtype=float)
                t_offsets  = np.asarray(waves.get("ECG_T_Offsets", []), dtype=float)
                r_peaks_arr= np.asarray(rpeaks["ECG_R_Peaks"],          dtype=float)

                n_beats = min(
                    len(p_onsets), len(q_peaks),
                    len(s_offsets), len(t_offsets),
                    len(r_peaks_arr),
                )
                for b in range(n_beats):
                    p_on  = p_onsets[b]
                    q_pk  = q_peaks[b]
                    s_off = s_offsets[b]
                    t_off = t_offsets[b]
                    r_pk  = r_peaks_arr[b]

                    if any(np.isnan([p_on, q_pk, s_off, t_off, r_pk])):
                        continue

                    fs = self.fs
                    pr  = (q_pk  - p_on)  / fs * 1000.0
                    qrs = (s_off - q_pk)  / fs * 1000.0
                    qt  = (t_off - q_pk)  / fs * 1000.0

                    # RR в мс для Bazett
                    if b > 0:
                        rr_ms = (r_pk - r_peaks_arr[b - 1]) / fs * 1000.0
                    elif b < n_beats - 1:
                        rr_ms = (r_peaks_arr[b + 1] - r_pk) / fs * 1000.0
                    else:
                        rr_ms = float("nan")

                    if (
                        pr  > 0 and qrs > 0 and qt > 0
                        and not np.isnan(rr_ms) and rr_ms > 0
                    ):
                        qtc = qt / np.sqrt(rr_ms / 1000.0)   # Bazett correction
                        pr_list.append(pr)
                        qrs_list.append(qrs)
                        qt_list.append(qt)
                        qtc_list.append(qtc)

            except Exception:
                pass  # delineation có thể thất bại trên signal quá nhiễu

        def _safe_mean(lst: list) -> float:
            return float(np.mean(lst)) if lst else float("nan")

        return {
            "pr_interval_ms":  _safe_mean(pr_list),
            "qrs_duration_ms": _safe_mean(qrs_list),
            "qt_interval_ms":  _safe_mean(qt_list),
            "qtc_interval_ms": _safe_mean(qtc_list),
        }
