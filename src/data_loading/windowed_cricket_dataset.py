from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .load_visig import ViSigSample, to_flat_sequence
except ImportError:
    # Fallback for when running as a script
    import sys
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from load_visig import ViSigSample, to_flat_sequence


class WindowedCricketDataset(Dataset):
    """
    Dataset that generates fixed-length overlapping windows from ViSig samples.

    Each item:
      - x: FloatTensor (window_size, feature_dim)
      - y: LongTensor scalar (class index, inherited from whole trial)
    """

    def __init__(
        self,
        samples: List[ViSigSample],
        label_to_idx: dict,
        window_size: int = 32,
        stride: int = 8,
        pad_value: float = 0.0,
        use_upper_tri_dist: bool = True,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,  
    ) -> None:
        if not samples:
            raise ValueError("Cannot create dataset from empty samples list")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.samples = samples
        self.label_to_idx = label_to_idx
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.pad_value = pad_value
        self.use_upper_tri_dist = use_upper_tri_dist
        self.transform = transform

        first = to_flat_sequence(samples[0], use_upper_tri_dist=self.use_upper_tri_dist)
        if first.ndim != 2:
            raise ValueError(f"Expected (T, F) array from to_flat_sequence, got {first.shape}")
        self.feature_dim = int(first.shape[1])

        self._trial_label_indices: List[int] = [self.label_to_idx[s.label] for s in samples]

        self._index: List[Tuple[int, int]] = []
        for si, s in enumerate(samples):
            seq = to_flat_sequence(s, use_upper_tri_dist=self.use_upper_tri_dist)
            T = int(seq.shape[0])
            W = self.window_size
            if T <= 0:
                self._index.append((si, 0))
                continue

            if T < W:
                self._index.append((si, 0))
                continue

            for st in range(0, T - W + 1, self.stride):
                self._index.append((si, st))
            last_start = T - W
            if len(self._index) == 0 or self._index[-1] != (si, last_start):
                if last_start > 0:
                    self._index.append((si, last_start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        si, st = self._index[idx]
        s = self.samples[si]
        seq = to_flat_sequence(s, use_upper_tri_dist=self.use_upper_tri_dist) 
        T, F = seq.shape
        W = self.window_size

        if T >= st + W:
            win = seq[st:st + W, :]
        else:
            needed = st + W - T
            pad = np.full((needed, F), self.pad_value, dtype=seq.dtype)
            tail = seq[st:T, :]
            win = np.vstack([tail, pad])

        if self.transform is not None:
            win = self.transform(win)

        x = torch.from_numpy(win).float() 
        try:
            y = torch.tensor(self.label_to_idx[s.label], dtype=torch.long)
        except KeyError as e:
            raise KeyError(f"Label '{s.label}' not found in label_to_idx") from e
        return x, y

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)

    @property
    def window_to_trial(self) -> List[int]:
        """
        Mapping from window index -> trial index (within self.samples).
        """
        return [si for (si, _) in self._index]

    @property
    def trial_labels(self) -> List[int]:
        """
        True label indices per trial index.
        """
        return self._trial_label_indices


