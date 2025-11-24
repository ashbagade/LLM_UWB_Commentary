from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ACC_SLICE = slice(0, 90)
GYRO_SLICE = slice(90, 180)
UWB_SLICE = slice(180, 195)


class Compose:
    def __init__(self, transforms: Sequence[Callable[[np.ndarray], np.ndarray]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x

    def __len__(self) -> int:
        return len(self.transforms)


class IMUZeroOffset:
    """
    Subtract per-window mean from IMU channels (acc + gyro).
    Operates on window/sequence shaped (T, F).
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"IMUZeroOffset expects (T, F), got {x.shape}")
        y = x.copy()
        imu_slice = slice(ACC_SLICE.start, GYRO_SLICE.stop)
        imu_mean = y[:, imu_slice].mean(axis=0, keepdims=True)
        y[:, imu_slice] = y[:, imu_slice] - imu_mean
        return y


class LowPassSmooth:
    """
    Simple moving average smoothing over time for IMU channels.
    kernel_size must be odd.
    """
    def __init__(self, kernel_size: int = 5) -> None:
        if kernel_size <= 1:
            self.kernel_size = 1
        else:
            self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"LowPassSmooth expects (T, F), got {x.shape}")
        if self.kernel_size <= 1:
            return x
        y = x.copy()
        k = self.kernel_size
        pad = k // 2
        imu_slice = slice(ACC_SLICE.start, GYRO_SLICE.stop)
        imu = y[:, imu_slice]
        padded = np.pad(imu, ((pad, pad), (0, 0)), mode="edge")
        cumsum = np.cumsum(padded, axis=0)
        cumsum = np.vstack([np.zeros((1, cumsum.shape[1]), dtype=cumsum.dtype), cumsum])  # (T+2p+1, F)
        smoothed = (cumsum[k:, :] - cumsum[:-k, :]) / float(k) 
        y[:, imu_slice] = smoothed
        return y


class UWBCorrectStaticClamp:
    """
    For windows that are approximately static (low accel variance),
    clamp UWB spikes toward a capped value (e.g., per-feature 75th percentile).
    """
    def __init__(self, accel_var_thresh: float = 1e-3, cap_percentile: float = 75.0) -> None:
        self.accel_var_thresh = float(accel_var_thresh)
        self.cap_percentile = float(cap_percentile)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"UWBCorrectStaticClamp expects (T, F), got {x.shape}")
        y = x.copy()
        acc = y[:, ACC_SLICE]
        var = acc.var(axis=0).mean()
        if var < self.accel_var_thresh:
            uwb = y[:, UWB_SLICE]
            cap = np.percentile(uwb, self.cap_percentile, axis=0, keepdims=True)
            uwb = np.minimum(uwb, cap)
            y[:, UWB_SLICE] = uwb
        return y


@dataclass
class StandardizeFeatures:
    """
    Per-feature standardization: (x - mean) / std.
    Mean/std must be provided or fitted via `fit_from_dataset`.
    """
    mean: Optional[np.ndarray] = None  
    std: Optional[np.ndarray] = None   
    eps: float = 1e-6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"StandardizeFeatures expects (T, F), got {x.shape}")
        if self.mean is None or self.std is None:
            raise ValueError("StandardizeFeatures not fitted: mean/std is None")
        return (x - self.mean.reshape(1, -1)) / (self.std.reshape(1, -1) + self.eps)

    @staticmethod
    def compute_mean_std_over_dataset(
        dataset,
        pre_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-feature mean and std across all timepoints of all items in dataset.
        The dataset must yield (x, y) with x as torch.Tensor of shape (T,F).
        """
        sum_vec = None
        sumsq_vec = None
        count = 0
        feature_dim = None

        for i in range(len(dataset)):
            x, _ = dataset[i]
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = np.asarray(x, dtype=np.float32)

            if pre_transform is not None:
                x_np = pre_transform(x_np)

            if x_np.ndim != 2:
                raise ValueError(f"Expected (T, F) from dataset item, got {x_np.shape}")
            T, F = x_np.shape
            if feature_dim is None:
                feature_dim = F
                sum_vec = np.zeros(F, dtype=np.float64)
                sumsq_vec = np.zeros(F, dtype=np.float64)
            elif F != feature_dim:
                raise ValueError(f"Inconsistent feature dims: expected {feature_dim}, got {F}")

            sum_vec += x_np.sum(axis=0)
            sumsq_vec += (x_np ** 2).sum(axis=0)
            count += T

        if count == 0 or sum_vec is None or sumsq_vec is None:
            raise ValueError("Empty dataset encountered during mean/std computation")

        mean = sum_vec / float(count)
        var = sumsq_vec / float(count) - mean**2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        std = np.where(std < 1e-8, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean, "std": self.std}, str(path))

    @staticmethod
    def load(path: Path) -> "StandardizeFeatures":
        data = torch.load(str(path), map_location="cpu")
        return StandardizeFeatures(mean=data["mean"], std=data["std"])


