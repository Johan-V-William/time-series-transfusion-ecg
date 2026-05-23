from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SegmentationConfig:
    K: int = 5
    eta_s: float = 0.2
    eta_of: float = 0.8
    p_b: float = 0.9
    eta_w: float = 3.0
    eta_cmin: float = 0.6
    eta_cmax: float = 1.4
    eta_delta: float = 0.2
    eta_ar: float = 0.4
    eta_l: float = 0.4
    eta_r: float = 0.6
    eta_sof: float = 0.25


@dataclass(frozen=True)
class SignalConstants:
    fs: int = 360
    M: int = 512
    max_seg_sec: float = 512 / 360


@dataclass(frozen=True)
class AugmentationConstants:
    shifts: tuple = (0.04, 0.08, 0.12)
    trims: tuple = (0.08, 0.16, 0.20, 0.24, 0.30)
    n_positive: int = 12
    n_negative: int = 12


@dataclass
class TrainingConfig:
    n_folds: int = 10
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    optimizer: str = "adam"
    dropout: float = 0.5
    weight_decay: float = 1e-4
    num_records_for_test: Optional[int] = None
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints/")
    log_dir: Path = Path("logs/")


@dataclass
class CNNConfig:
    in_channels: int = 1
    conv_channels: tuple = (32, 64, 128, 128, 64)
    kernel_sizes: tuple = (5, 5, 3, 3, 3)
    pool_size: int = 2
    fc_hidden: int = 128
    n_classes: int = 2
    dropout: float = 0.5
    input_length: int = 512


@dataclass
class Config:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    signal: SignalConstants = field(default_factory=SignalConstants)
    augment: AugmentationConstants = field(default_factory=AugmentationConstants)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)


CFG = Config()
