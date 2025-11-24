from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Union

import numpy as np
from scipy.io import loadmat

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


@dataclass
class ViSigSample:
    """Represents a single ViSig cricket umpire signal trial."""
    file_path: Path
    file_stem: str          
    label: str              
    participant_id: Optional[int] 
    acc: np.ndarray     
    gyro: np.ndarray        
    dist: np.ndarray       
    t: np.ndarray          


def infer_label_and_participant(stem: str) -> Tuple[str, Optional[int]]:
    """
    Given a filename stem like 'boundary4_1', return ('boundary4', 1).
    If no numeric suffix is present, return (stem, None).
    
    Args:
        stem: Filename without extension, e.g. 'boundary4_1' or 'noball_5'
    
    Returns:
        Tuple of (label, participant_id). participant_id is None if no numeric suffix.
    
    Examples:
        >>> infer_label_and_participant('boundary4_1')
        ('boundary4', 1)
        >>> infer_label_and_participant('noball_5')
        ('noball', 5)
        >>> infer_label_and_participant('wide')
        ('wide', None)
    """
    if '_' not in stem:
        return stem, None
    
    parts = stem.rsplit('_', 1)
    if len(parts) != 2:
        return stem, None
    
    label_part, suffix_part = parts
    try:
        participant_id = int(suffix_part)
        return label_part, participant_id
    except ValueError:
        return stem, None


def load_visig_mat(path: Union[str, Path]) -> ViSigSample:
    """
    Load a single ViSig .mat file and return a ViSigSample.
    Ensures arrays are converted to time-major format and validated.
    
    Args:
        path: Path to the .mat file
    
    Returns:
        ViSigSample with normalized time-major arrays
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If required keys are missing from the .mat file
        ValueError: If array shapes are inconsistent or unexpected
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    
    required_keys = ['acc_mat', 'gyro_mat', 'dist_mat', 'rawt']
    missing_keys = [key for key in required_keys if key not in mat]
    if missing_keys:
        raise KeyError(
            f"Missing required keys in {path}: {missing_keys}. "
            f"Available keys: {list(mat.keys())}"
        )
    
    acc = np.asarray(mat['acc_mat'])
    if acc.ndim < 3:
        raise ValueError(
            f"acc_mat in {path} has unexpected ndim={acc.ndim}, expected at least 3"
        )
    acc = np.squeeze(acc)  
    if acc.ndim != 3:
        raise ValueError(
            f"acc_mat in {path} has unexpected shape after squeeze: {acc.shape}, "
            f"expected 3D array"
        )
    if acc.shape[0] != 6 or acc.shape[1] != 15:
        raise ValueError(
            f"acc_mat in {path} has unexpected shape: {acc.shape}, "
            f"expected (6, 15, N). Got first two dims: ({acc.shape[0]}, {acc.shape[1]})"
        )
    acc = acc.transpose(2, 0, 1)  
    
    gyro = np.asarray(mat['gyro_mat'])
    if gyro.ndim < 3:
        raise ValueError(
            f"gyro_mat in {path} has unexpected ndim={gyro.ndim}, expected at least 3"
        )
    gyro = np.squeeze(gyro)
    if gyro.ndim != 3:
        raise ValueError(
            f"gyro_mat in {path} has unexpected shape after squeeze: {gyro.shape}, "
            f"expected 3D array"
        )
    if gyro.shape[0] != 6 or gyro.shape[1] != 15:
        raise ValueError(
            f"gyro_mat in {path} has unexpected shape: {gyro.shape}, "
            f"expected (6, 15, N). Got first two dims: ({gyro.shape[0]}, {gyro.shape[1]})"
        )
    gyro = gyro.transpose(2, 0, 1)  
    
    dist = np.asarray(mat['dist_mat'])
    if dist.ndim < 3:
        raise ValueError(
            f"dist_mat in {path} has unexpected ndim={dist.ndim}, expected at least 3"
        )
    dist = np.squeeze(dist)
    if dist.ndim != 3:
        raise ValueError(
            f"dist_mat in {path} has unexpected shape after squeeze: {dist.shape}, "
            f"expected 3D array"
        )
    if dist.shape[0] != 6 or dist.shape[1] != 6:
        raise ValueError(
            f"dist_mat in {path} has unexpected shape: {dist.shape}, "
            f"expected (6, 6, N). Got first two dims: ({dist.shape[0]}, {dist.shape[1]})"
        )
    dist = dist.transpose(2, 0, 1)  
    
    rawt = np.asarray(mat['rawt'])
    rawt = np.squeeze(rawt)
    if rawt.ndim != 1:
        raise ValueError(
            f"rawt in {path} has unexpected shape after squeeze: {rawt.shape}, "
            f"expected 1D array"
        )
    rawt = rawt.reshape(-1).astype(float)
    
    T = acc.shape[0]
    if gyro.shape[0] != T:
        raise ValueError(
            f"Time dimension mismatch in {path}: acc has T={T}, gyro has T={gyro.shape[0]}"
        )
    if dist.shape[0] != T:
        raise ValueError(
            f"Time dimension mismatch in {path}: acc has T={T}, dist has T={dist.shape[0]}"
        )
    if rawt.shape[0] != T:
        raise ValueError(
            f"Time dimension mismatch in {path}: acc has T={T}, rawt has T={rawt.shape[0]}"
        )
    
    file_stem = path.stem
    label, participant_id = infer_label_and_participant(file_stem)
    
    return ViSigSample(
        file_path=path,
        file_stem=file_stem,
        label=label,
        participant_id=participant_id,
        acc=acc,
        gyro=gyro,
        dist=dist,
        t=rawt,
    )


def load_visig_dataset(
    root: Union[str, Path],
    pattern: str = "*.mat",
    allowed_labels: Optional[Iterable[str]] = None,
) -> List[ViSigSample]:
    """
    Recursively load all .mat files under `root` matching `pattern`.
    Optionally filter by `allowed_labels`.
    Returns a list[ViSigSample], sorted by file path for determinism.
    
    Args:
        root: Root directory to search for .mat files
        pattern: Glob pattern to match files (default: "*.mat")
        allowed_labels: Optional iterable of labels to include. If None, include all.
    
    Returns:
        List of ViSigSample objects, sorted by file_path
    
    Raises:
        RuntimeError: If no files are found
        FileNotFoundError: If root directory doesn't exist
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    
    if not root.is_dir():
        raise ValueError(f"Root must be a directory: {root}")
    
    allowed_set = None
    if allowed_labels is not None:
        allowed_set = set(allowed_labels)
    
    mat_files = list(root.rglob(pattern))
    mat_files = [f for f in mat_files if f.is_file()]
    
    if not mat_files:
        raise RuntimeError(
            f"No .mat files found under {root} matching pattern '{pattern}'"
        )
    
    samples = []
    for mat_file in mat_files:
        try:
            sample = load_visig_mat(mat_file)
            if allowed_set is None or sample.label in allowed_set:
                samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to load {mat_file}: {e}")
            continue
    
    if not samples:
        raise RuntimeError(
            f"No valid samples loaded from {root}. "
            f"Found {len(mat_files)} files but none matched filters or loaded successfully."
        )
    
    samples.sort(key=lambda s: s.file_path)
    
    return samples


def get_label_distribution(samples: List[ViSigSample]) -> Dict[str, int]:
    """
    Return a dict mapping label -> count, useful for quick EDA.
    
    Args:
        samples: List of ViSigSample objects
    
    Returns:
        Dictionary mapping label strings to counts
    
    Example:
        >>> samples = load_visig_dataset("path/to/data")
        >>> dist = get_label_distribution(samples)
        >>> print(dist)
        {'boundary4': 8, 'noball': 8, 'wide': 8, ...}
    """
    distribution: Dict[str, int] = {}
    for sample in samples:
        distribution[sample.label] = distribution.get(sample.label, 0) + 1
    return distribution


def to_flat_sequence(
    sample: ViSigSample,
    use_upper_tri_dist: bool = True,
) -> np.ndarray:
    """
    Convert a ViSigSample into a (T, F) feature matrix.
    
    - Flattens acc (6,15) and gyro (6,15) for each timestep.
    - For dist:
      - if use_upper_tri_dist: use upper triangle (excluding diagonal) of 6x6
      - else: use full 6x6.
    
    Args:
        sample: ViSigSample to convert
        use_upper_tri_dist: If True, use upper triangle of distance matrix (excluding diagonal).
                           If False, use full 6x6 distance matrix.
    
    Returns:
        seq: np.ndarray of shape (T, F) where:
            - T is the number of time steps
            - F is the feature dimension (varies based on use_upper_tri_dist)
    
    Example:
        >>> sample = load_visig_mat("path/to/file.mat")
        >>> features = to_flat_sequence(sample)
        >>> print(features.shape)  # (T, F)
    """
    T = sample.acc.shape[0]
    
    acc_flat = sample.acc.reshape(T, -1)  # (T, 90)
    gyro_flat = sample.gyro.reshape(T, -1)  # (T, 90)
    
    if use_upper_tri_dist:
        triu_indices = np.triu_indices(6, k=1)
        dist_flat = sample.dist[:, triu_indices[0], triu_indices[1]]  # (T, 15)
    else:
        dist_flat = sample.dist.reshape(T, -1)  # (T, 36)
    
    seq = np.concatenate([acc_flat, gyro_flat, dist_flat], axis=1)
    
    return seq


if __name__ == "__main__":
    import os
    
    if load_dotenv is not None:
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
    
    visig_root = os.getenv("VISIG_ROOT")
    if visig_root:
        print(f"Loading ViSig dataset from: {visig_root}")
        try:
            samples = load_visig_dataset(visig_root)
            print(f"\nLoaded {len(samples)} samples")
            
            label_dist = get_label_distribution(samples)
            print(f"\nLabel distribution:")
            for label, count in sorted(label_dist.items()):
                print(f"  {label}: {count}")
            
            if samples:
                first_sample = samples[0]
                flat_seq = to_flat_sequence(first_sample)
                print(f"\nFirst sample:")
                print(f"  File: {first_sample.file_path.name}")
                print(f"  Label: {first_sample.label}")
                print(f"  Participant ID: {first_sample.participant_id}")
                print(f"  Time steps: {first_sample.acc.shape[0]}")
                print(f"  Flat sequence shape: {flat_seq.shape}")
                print(f"  Feature breakdown:")
                print(f"    - acc: {first_sample.acc.shape[1] * first_sample.acc.shape[2]} features")
                print(f"    - gyro: {first_sample.gyro.shape[1] * first_sample.gyro.shape[2]} features")
                print(f"    - dist (upper tri): {flat_seq.shape[1] - 180} features")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Set VISIG_ROOT environment variable to test data loading")
        print("Example: export VISIG_ROOT=/path/to/visig_body_signal_data/data/cricket")

