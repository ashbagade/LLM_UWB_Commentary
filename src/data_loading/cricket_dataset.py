from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Union
from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

try:
    from .load_visig import (
        ViSigSample,
        load_visig_dataset,
        to_flat_sequence,
        get_label_distribution,
    )
except ImportError:
    # Fallback for when running as a script
    import sys
    # Add parent directory to path so we can import load_visig
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from load_visig import (
        ViSigSample,
        load_visig_dataset,
        to_flat_sequence,
        get_label_distribution,
    )


def build_label_mapping(samples: List[ViSigSample]) -> Dict[str, int]:
    """
    Build a stable mapping from label string to integer index.

    Labels are sorted lexicographically for determinism.

    Args:
        samples: List of ViSigSample objects

    Returns:
        Dictionary mapping label strings to integer indices (0-indexed)

    Raises:
        ValueError: If samples list is empty

    Example:
        >>> samples = [ViSigSample(...), ...]
        >>> mapping = build_label_mapping(samples)
        >>> print(mapping)
        {'boundary4': 0, 'noball': 1, 'wide': 2, ...}
    """
    if not samples:
        raise ValueError("Cannot build label mapping from empty samples list")
    
    unique_labels = sorted(set(sample.label for sample in samples))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_idx


class CricketSignalsDataset(Dataset):
    """
    PyTorch Dataset for ViSig cricket umpire sensor data.

    Each item is:
        x: FloatTensor of shape (max_len, feature_dim)
        y: LongTensor scalar (class index)

    Uses `to_flat_sequence` to convert each ViSigSample into (T, F),
    then applies center-cropping or end-padding to reach a fixed length.
    """

    def __init__(
        self,
        samples: List[ViSigSample],
        label_to_idx: Optional[Dict[str, int]] = None,
        max_len: int = 400,
        use_upper_tri_dist: bool = True,
        pad_value: float = 0.0,
    ):
        """
        Initialize the dataset.

        Args:
            samples: List of ViSigSample objects to include in the dataset
            label_to_idx: Optional pre-computed label mapping. If None, will be built from samples.
            max_len: Maximum sequence length. Sequences longer than this will be center-cropped,
                    shorter sequences will be end-padded.
            use_upper_tri_dist: Whether to use upper triangle of distance matrix in feature extraction
            pad_value: Value to use for padding shorter sequences

        Raises:
            ValueError: If samples list is empty or feature dimension inference fails
        """
        if not samples:
            raise ValueError("Cannot create dataset from empty samples list")
        
        self.samples = samples
        self.max_len = max_len
        self.use_upper_tri_dist = use_upper_tri_dist
        self.pad_value = pad_value
        
        if label_to_idx is None:
            self.label_to_idx = build_label_mapping(samples)
        else:
            self.label_to_idx = label_to_idx
        
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        try:
            first_seq = to_flat_sequence(samples[0], use_upper_tri_dist=self.use_upper_tri_dist)
            if first_seq.ndim != 2:
                raise ValueError(
                    f"Expected 2D sequence from to_flat_sequence, got shape {first_seq.shape}"
                )
            self.feature_dim = first_seq.shape[1]
        except Exception as e:
            raise ValueError(
                f"Failed to infer feature dimension from first sample: {e}"
            ) from e

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (x, y) where:
                x: FloatTensor of shape (max_len, feature_dim)
                y: LongTensor scalar (class index)

        Raises:
            ValueError: If feature dimension mismatch or sequence shape is unexpected
            KeyError: If sample label is not in label_to_idx
        """
        sample = self.samples[idx]

        seq = to_flat_sequence(sample, use_upper_tri_dist=self.use_upper_tri_dist)
        if seq.ndim != 2:
            raise ValueError(
                f"Expected (T, F) sequence, got shape {seq.shape} for {sample.file_path}"
            )

        T, F = seq.shape
        if F != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch for {sample.file_path}: "
                f"got F={F}, expected {self.feature_dim}"
            )

        max_len = self.max_len
        if T > max_len:
            start = (T - max_len) // 2
            seq = seq[start:start + max_len, :]
        elif T < max_len:
            pad_rows = max_len - T
            pad_block = np.full((pad_rows, F), self.pad_value, dtype=seq.dtype)
            seq = np.vstack([seq, pad_block])

        try:
            y_idx = self.label_to_idx[sample.label]
        except KeyError as e:
            raise KeyError(
                f"Label '{sample.label}' not found in label_to_idx"
            ) from e

        x_tensor = torch.from_numpy(seq).float()          
        y_tensor = torch.tensor(y_idx, dtype=torch.long)  

        return x_tensor, y_tensor

    @property
    def num_classes(self) -> int:
        """Return the number of unique classes in the dataset."""
        return len(self.label_to_idx)


def create_cricket_datasets(
    root: Union[str, Path],
    pattern: str = "*.mat",
    allowed_labels: Optional[Iterable[str]] = None,
    max_len: int = 400,
    use_upper_tri_dist: bool = True,
    pad_value: float = 0.0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[CricketSignalsDataset, CricketSignalsDataset, CricketSignalsDataset]:
    """
    Load ViSig samples from disk and split into train/val/test `CricketSignalsDataset`s.

    Currently performs a random split over samples.
    Later we may replace this with subject-wise or stratified splitting.

    Args:
        root: Root directory containing .mat files
        pattern: Glob pattern to match files (default: "*.mat")
        allowed_labels: Optional iterable of labels to include. If None, include all.
        max_len: Maximum sequence length for padding/truncation
        use_upper_tri_dist: Whether to use upper triangle of distance matrix
        pad_value: Value to use for padding shorter sequences
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        seed: Random seed for reproducible splits (default: 42)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Raises:
        RuntimeError: If no samples are loaded
        ValueError: If split ratios are invalid or result in empty splits
    """
    samples = load_visig_dataset(root, pattern=pattern, allowed_labels=allowed_labels)
    if not samples:
        raise RuntimeError(f"No samples loaded from {root}")

    label_to_idx = build_label_mapping(samples)

    full_ds = CricketSignalsDataset(
        samples=samples,
        label_to_idx=label_to_idx,
        max_len=max_len,
        use_upper_tri_dist=use_upper_tri_dist,
        pad_value=pad_value,
    )

    n_total = len(full_ds)
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be between 0 and 1")

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes derived from ratios: "
            f"train={n_train}, val={n_val}, test={n_test}, total={n_total}"
        )

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=g
    )

    return train_ds, val_ds, test_ds


def stratified_trial_split(
    samples: List[ViSigSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ViSigSample], List[ViSigSample], List[ViSigSample]]:
    """
    Split samples into train/val/test in a label-stratified way at the TRIAL level.
    Ensures each label appears in all splits when possible.
    """
    if not samples:
        raise ValueError("No samples provided for stratified split.")

    by_label: Dict[str, List[ViSigSample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)

    rng = random.Random(seed)
    train: List[ViSigSample] = []
    val: List[ViSigSample] = []
    test: List[ViSigSample] = []

    for label, group in by_label.items():
        if len(group) < 3:
            raise ValueError(f"Not enough trials for label '{label}' to stratify: {len(group)} found.")

        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, n_train - 1)
            n_val = max(1, n_val - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    train.sort(key=lambda s: s.file_path)
    val.sort(key=lambda s: s.file_path)
    test.sort(key=lambda s: s.file_path)
    return train, val, test


def create_cricket_datasets_stratified(
    root: Union[str, Path],
    pattern: str = "*.mat",
    allowed_labels: Optional[Iterable[str]] = None,
    max_len: int = 400,
    use_upper_tri_dist: bool = True,
    pad_value: float = 0.0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[CricketSignalsDataset, CricketSignalsDataset, CricketSignalsDataset]:
    """
    Label-stratified variant of create_cricket_datasets.
    Ensures each class has representation in train/val/test.
    """
    samples = load_visig_dataset(root, pattern=pattern, allowed_labels=allowed_labels)
    if not samples:
        raise RuntimeError(f"No samples loaded from {root}")

    train_samples, val_samples, test_samples = stratified_trial_split(
        samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    label_to_idx = build_label_mapping(samples)

    train_ds = CricketSignalsDataset(
        samples=train_samples,
        label_to_idx=label_to_idx,
        max_len=max_len,
        use_upper_tri_dist=use_upper_tri_dist,
        pad_value=pad_value,
    )
    val_ds = CricketSignalsDataset(
        samples=val_samples,
        label_to_idx=label_to_idx,
        max_len=max_len,
        use_upper_tri_dist=use_upper_tri_dist,
        pad_value=pad_value,
    )
    test_ds = CricketSignalsDataset(
        samples=test_samples,
        label_to_idx=label_to_idx,
        max_len=max_len,
        use_upper_tri_dist=use_upper_tri_dist,
        pad_value=pad_value,
    )

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    import os
    
    # Load environment variables from .env file if dotenv is available
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
    except ImportError:
        pass  # dotenv not available, skip
    
    visig_root = os.getenv("VISIG_ROOT")
    if not visig_root:
        print("Set VISIG_ROOT to test dataset creation.")
    else:
        print(f"Loading datasets from {visig_root}")
        train_ds, val_ds, test_ds = create_cricket_datasets(visig_root)
        print(f"Train/Val/Test sizes: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}")
        example_x, example_y = train_ds[0]
        print(f"Example x shape: {example_x.shape}, dtype={example_x.dtype}")
        print(f"Example y: {example_y.item()}")

