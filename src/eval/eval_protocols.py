from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Dict
from pathlib import Path
from collections import defaultdict
import random

from src.data_loading.load_visig import ViSigSample, load_visig_dataset
from src.data_loading.cricket_dataset import stratified_trial_split


def make_random_split(
    root: str,
    pattern: str = "*.mat",
    allowed_labels: Optional[Iterable[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ViSigSample], List[ViSigSample], List[ViSigSample]]:
    samples = load_visig_dataset(root, pattern=pattern, allowed_labels=allowed_labels)
    return stratified_trial_split(samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


def make_lopo_splits(
    root: str,
    pattern: str = "*.mat",
    allowed_labels: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> List[Tuple[List[ViSigSample], List[ViSigSample], int]]:
    """
    Leave-One-Participant-Out splits.
    Returns a list of (train_samples, test_samples, heldout_pid).
    A small validation set can be carved from train_samples by the caller using stratified splitting.
    """
    samples = load_visig_dataset(root, pattern=pattern, allowed_labels=allowed_labels)
    by_pid: Dict[int, List[ViSigSample]] = defaultdict(list)
    for s in samples:
        if s.participant_id is None:
            continue
        by_pid[s.participant_id].append(s)

    pids = sorted(by_pid.keys())
    splits: List[Tuple[List[ViSigSample], List[ViSigSample], int]] = []
    for pid in pids:
        test_samples = by_pid[pid]
        train_samples = [s for s in samples if s.participant_id != pid]
        splits.append((train_samples, test_samples, pid))
    return splits


