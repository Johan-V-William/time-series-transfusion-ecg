"""
sigw1_metric.py

Module độc lập tính chỉ số Sig-W1 (Expected Signature RMSE) giữa hai phân phối chuỗi thời gian.
Sử dụng thư viện signatory để tính signature.
"""

import math
from typing import Optional, Tuple

import torch
import signatory


# ============================
# Các phép biến đổi tăng cường (augmentations) cơ bản
# ============================
class BaseAugmentation:
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Scale(BaseAugmentation):
    """Nhân toàn bộ path với một hệ số."""
    def __init__(self, scale: float = 1.0, dim: Optional[int] = None):
        self.scale = scale
        self.dim = dim

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return self.scale * x
        else:
            x[..., self.dim] = self.scale * x[..., self.dim]
            return x


class AddTime(BaseAugmentation):
    """Thêm kênh thời gian (chuẩn hóa về [0,1]) vào path."""
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        t = torch.linspace(0, 1, length, device=x.device).reshape(1, -1, 1).repeat(batch, 1, 1)
        return torch.cat([t, x], dim=-1)


class CumSum(BaseAugmentation):
    """Tích lũy (cumsum) theo thời gian."""
    def __init__(self, dim: int = 1):
        self.dim = dim

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x.cumsum(dim=self.dim)


def apply_augmentations(x: torch.Tensor, augmentations: Tuple[BaseAugmentation, ...]) -> torch.Tensor:
    """Áp dụng lần lượt các augmentation lên tensor đầu vào."""
    y = x.clone()
    for aug in augmentations:
        y = aug.apply(y)
    return y


# ============================
# Hàm tính Expected Signature
# ============================
def expected_signature(
    paths: torch.Tensor,
    depth: int,
    augmentations: Tuple[BaseAugmentation, ...] = (),
    normalise: bool = True
) -> torch.Tensor:
    """
    Tính kỳ vọng signature (trung bình theo batch) của tập các đường đi.

    Args:
        paths: (batch, time, channels)
        depth: bậc tối đa của signature (≥ 1)
        augmentations: danh sách các phép biến đổi áp dụng trước khi tính signature
        normalise: nếu True, nhân mỗi thành phần signature bậc k với k! (để chuẩn hóa)

    Returns:
        vector expected signature (shape: signature_dim,)
    """
    x = apply_augmentations(paths, augmentations)
    sig = signatory.signature(x, depth)          # (batch, signature_dim)
    exp_sig = sig.mean(dim=0)                    # (signature_dim,)

    if normalise:
        dim = x.shape[-1]
        count = 0
        for i in range(depth):
            block_size = dim ** (i + 1)
            exp_sig[count:count+block_size] *= math.factorial(i + 1)
            count += block_size
    return exp_sig


# ============================
# Lớp tính khoảng cách Sig-W1
# ============================
class SigW1Metric:
    """
    Sig-W1 metric = RMSE giữa expected signature của dữ liệu thực và dữ liệu sinh.
    Giá trị càng nhỏ càng tốt.
    """
    def __init__(
        self,
        real_data: torch.Tensor,
        depth: int,
        augmentations: Tuple[BaseAugmentation, ...] = (),
        normalise: bool = True
    ):
        """
        Args:
            real_data: dữ liệu thật, shape (N, T, C)
            depth: bậc signature
            augmentations: các augmentation áp dụng lên cả real và fake
            normalise: có chuẩn hóa factorial hay không
        """
        self.depth = depth
        self.augmentations = augmentations
        self.normalise = normalise
        self.real_expected_sig = expected_signature(
            real_data, depth, augmentations, normalise
        )

    def __call__(self, fake_data: torch.Tensor) -> torch.Tensor:
        """
        Tính Sig-W1 giữa fake_data và real_data (đã lưu từ __init__).

        Args:
            fake_data: dữ liệu sinh, shape (M, T, C)

        Returns:
            giá trị RMSE (scalar tensor)
        """
        fake_expected_sig = expected_signature(
            fake_data, self.depth, self.augmentations, self.normalise
        )
        diff = fake_expected_sig - self.real_expected_sig.to(fake_data.device)
        return torch.sqrt(torch.mean(diff ** 2))


# ============================
# Ví dụ sử dụng
# ============================
if __name__ == "__main__":
    # Tạo dữ liệu giả
    real = torch.randn(100, 20, 3)
    fake = torch.randn(100, 20, 3)

    # Không augmentation
    metric = SigW1Metric(real, depth=3, augmentations=(), normalise=True)
    loss = metric(fake)
    print(f"Sig-W1 (no aug): {loss.item():.6f}")

    # Có augmentation: thêm time và scale
    augs = (AddTime(), Scale(0.5))
    metric_aug = SigW1Metric(real, depth=3, augmentations=augs, normalise=True)
    loss_aug = metric_aug(fake)
    print(f"Sig-W1 (with aug): {loss_aug.item():.6f}")
