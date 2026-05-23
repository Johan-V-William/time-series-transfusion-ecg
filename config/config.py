from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional # Import Optional

@dataclass(frozen=True)
class MITConfig:
    """Configuration for MIT-BIH Arrhythmia dataset."""

    num_classes: int = 5
    class_names: list[str] = field(
        default_factory=lambda: ["N", "S", "V", "F", "Q"]
    )
    sequence_len: int = 186




@dataclass(frozen=True)
class SegmentationConfig: # New config for the adaptive window segmenter
    """Constants related to adaptive window segmentation algorithm (Alg 1 & 2)."""
    K:             int   = 5     # Number of beats for initialization (Alg 1)
    eta_s:         float = 0.2   # Step size factor (s = eta_s * C_bar)
    eta_of:        float = 0.8   # Omega adjustment factor for window start (wst = cp_abs + eta_of * omega)
    p_b:           float = 0.9   # CNN probability threshold for a valid beat
    eta_w:         float = 3.0   # Window size factor (omega = eta_w * C_bar)
    eta_cmin:      float = 0.6   # Min RR interval factor (cp_min = eta_cmin * C_bar)
    eta_cmax:      float = 1.4   # Max RR interval factor (cp_max = eta_cmax * C_bar)
    eta_delta:     float = 0.2   # Factor for Method I boundary width
    eta_ar:        float = 0.4   # Asymmetry ratio for Method I (0.4 for left, 0.6 for right)
    eta_l:         float = 0.4   # Factor for Method II left boundary
    eta_r:         float = 0.6   # Factor for Method II right boundary
    eta_sof:       float = 0.25  # Step on failure factor (when no beat detected)


@dataclass(frozen=True)
class SignalConstants:
    fs:          int = 360           # MIT-BIH sampling rate (Hz)
    M:           int = 512           # fixed CNN input length (samples)
    max_seg_sec: float = 512 / 360  # ~1.422s


# -- Augmentation constants -------------------------------------------------
@dataclass(frozen=True)
class AugmentationConstants:
    # Positive: shifts as fraction of beat length s
    shifts: tuple = (0.04, 0.08, 0.12)          # left AND right
    # Positive: trim fractions around main cp (from left edge) \
    trims:  tuple = (0.08, 0.16, 0.20, 0.24, 0.30)  # trimmed from RIGHT of window
    n_positive: int = 12   # 1 main + 6 shifted + 5 trimmed
    n_negative: int = 12


# -- Training hyperparameters -----------------------------------------------
@dataclass
class TrainingConfig:
    n_folds:    int   = 10
    epochs:     int   = 15
    batch_size: int   = 256
    lr:         float = 1e-3          # NOT in paper -- choose via ablation
    optimizer:  str   = "adam"
    dropout:    float = 0.5           # NOT in paper
    weight_decay: float = 1e-4
    num_records_for_test: Optional[int] = None # Added for quick testing

    # Paths
    data_dir:     Path = Path("/content/drive/MyDrive/mit-bih/data/mitdb") # Updated path
    checkpoint_dir: Path = Path("checkpoints/")
    log_dir:      Path = Path("logs/")


# -- CNN architecture -------------------------------------------------------
@dataclass
class CNNConfig:
    """

    """
    in_channels: int = 1
    conv_channels: tuple = (32, 64, 128, 128, 64)   # 5 layers
    kernel_sizes:  tuple = (5, 5, 3, 3, 3)
    pool_size:     int = 2
    fc_hidden:     int = 128
    n_classes:     int = 2    # VB=1, NVB=0
    dropout:       float = 0.5
    input_length:  int = 512

@dataclass
class TransformConfig:
    
    input_channels: int = 1
    kernel_size: int = 8
    stride: int = 1
    dropout: float = 0.2

    # Feature extractor
    mid_channels: int = 32
    final_out_channels: int = 128

    # Transformer
    trans_dim: int = 25
    num_heads: int = 5
    
# -- Tranform architecture -------------------------------------------------------
@dataclass
class TranformConfig:
    """
    
    """
    num_epochs: int = 60
    batch_size: int = 128
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    feature_dim: int = 128


# -- Master config ----------------------------------------------------------
@dataclass
class Config:
    segmentation: SegmentationConfig   = field(default_factory=SegmentationConfig)
    signal:      SignalConstants      = field(default_factory=SignalConstants)
    augment:     AugmentationConstants = field(default_factory=AugmentationConstants)
    training:    TrainingConfig       = field(default_factory=TrainingConfig)
    cnn:         CNNConfig            = field(default_factory=CNNConfig)
    tranform:    Tranform = field(default_factory=Tranform)


CFG = Config()
