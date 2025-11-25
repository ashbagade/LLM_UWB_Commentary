from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from src.data_loading.load_visig import load_visig_mat, to_flat_sequence
from src.inference.load_model import load_model_from_checkpoint, get_device


def make_windows(
    seq: np.ndarray,
    timestamps: Optional[np.ndarray],
    window_size: int,
    stride: int,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    Convert a (T, F) sequence into overlapping windows.

    Parameters
    ----------
    seq : np.ndarray
        Shape (T, F), time-major flat sequence.
    timestamps : np.ndarray or None
        Shape (T,). If provided, used to compute a representative time
        for each window (mean over the window). If None, synthetic indices
        are used instead.
    window_size : int
        Number of timesteps per window.
    stride : int
        Hop size between consecutive windows.
    pad_value : float, default 0.0
        Value used to pad the last window if needed.

    Returns
    -------
    windows : np.ndarray
        Array of shape (N, window_size, F) where N is number of windows.
    window_times : list of float
        Representative timestamp for each window.
    """
    assert seq.ndim == 2, f"Expected (T, F) array, got shape {seq.shape}"

    T, Fdim = seq.shape
    if T == 0:
        raise ValueError("Empty sequence: T == 0")

    if timestamps is not None:
        timestamps = np.asarray(timestamps).reshape(-1)
        if len(timestamps) != T:
            raise ValueError(
                f"timestamps length ({len(timestamps)}) does not match seq length ({T})"
            )

    windows: List[np.ndarray] = []
    window_times: List[float] = []

    start = 0
    while start < T:
        end = start + window_size
        if start >= T:
            break

        # Slice window; pad if needed
        win = seq[start:end]
        if win.shape[0] < window_size:
            pad_len = window_size - win.shape[0]
            pad = np.full((pad_len, Fdim), pad_value, dtype=seq.dtype)
            win = np.vstack([win, pad])

        windows.append(win)

        # Representative time = mean of timestamps in [start:end]
        if timestamps is not None:
            slice_ts = timestamps[start:min(end, T)]
            if slice_ts.size == 0:
                t_rep = float(timestamps[min(start, T - 1)])
            else:
                t_rep = float(slice_ts.mean())
        else:
            # Fallback to synthetic "time index"
            t_rep = float(start + min(window_size, T - start) / 2.0)

        window_times.append(t_rep)
        start += stride

    windows_arr = np.stack(windows, axis=0)  # (N, window_size, F)
    return windows_arr, window_times


def load_label_mapping(label_map_path: Optional[str]) -> Optional[Dict[int, str]]:
    """
    Load an optional label mapping JSON file.

    Supports two formats:

    1) label_to_idx:
        {
          "boundary4": 0,
          "boundary6": 1,
          ...
        }

       -> we invert it to idx_to_label.

    2) idx_to_label:
        {
          "0": "boundary4",
          "1": "boundary6",
          ...
        }

    Returns
    -------
    idx_to_label : dict[int, str] or None
    """
    if label_map_path is None:
        return None

    path = Path(label_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Label map file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    idx_to_label: Dict[int, str] = {}

    # Detect format
    if all(isinstance(v, int) for v in obj.values()):
        # label_to_idx: invert
        for label, idx in obj.items():
            idx_to_label[int(idx)] = label
    else:
        # idx_to_label: keys are indices as strings
        for k, v in obj.items():
            idx_to_label[int(k)] = str(v)

    return idx_to_label


def run_inference_on_mat(
    mat_path: str,
    checkpoint_path: str,
    window_size: int,
    stride: int,
    label_map_path: Optional[str] = None,
    device_str: Optional[str] = None,
    pad_value: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Core logic for running window-level inference on a single .mat file.

    Returns a list of prediction dicts:
        {
            "window_index": int,
            "time": float,
            "pred_idx": int,
            "pred_label": str or null,
            "prob": float
        }
    """
    # 1) Load sample using existing ViSig loader
    sample = load_visig_mat(mat_path)

    # 2) Convert to flat (T, F) sequence (same as training)
    seq = to_flat_sequence(sample, use_upper_tri_dist=True)  # (T, F)
    T, Fdim = seq.shape

    # 3) Get timestamps if available on the sample
    timestamps = getattr(sample, "t", None)

    # 4) Make windows
    windows_np, window_times = make_windows(
        seq=seq,
        timestamps=timestamps,
        window_size=window_size,
        stride=stride,
        pad_value=pad_value,
    )

    # 5) Build and load the model (metadata comes from checkpoint)
    device = get_device(device_str)
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # 6) Run inference
    windows_tensor = torch.from_numpy(windows_np).float().to(device)  # (N, L, F)
    with torch.no_grad():
        logits = model(windows_tensor)  # (N, num_classes)
        probs = F.softmax(logits, dim=1)  # (N, num_classes)
        pred_probs, pred_indices = probs.max(dim=1)  # (N,), (N,)

    idx_to_label = load_label_mapping(label_map_path)

    # 7) Build result objects
    results: List[Dict[str, Any]] = []
    for i in range(windows_np.shape[0]):
        idx = int(pred_indices[i].item())
        prob = float(pred_probs[i].item())
        t = float(window_times[i])

        if idx_to_label is not None:
            label_name = idx_to_label.get(idx)
        else:
            label_name = None

        results.append(
            {
                "window_index": i,
                "time": t,
                "pred_idx": idx,
                "pred_label": label_name,
                "prob": prob,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained CNN/LSTM sequence classifier on a single ViSig .mat trial "
            "and output window-level predictions."
        )
    )
    parser.add_argument(
        "--mat-path",
        type=str,
        required=True,
        help="Path to a single ViSig .mat file (one trial).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt as saved by train_seq_classifier).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="Number of timesteps per window (must match training).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Hop size between windows (must match training).",
    )
    parser.add_argument(
        "--label-map-json",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON label mapping. "
            "Supports either label_to_idx or idx_to_label format."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override: 'cuda', 'mps', or 'cpu'.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write predictions as JSON. "
             "If not provided, prints the first few predictions only.",
    )
    parser.add_argument(
        "--pad-value",
        type=float,
        default=0.0,
        help="Pad value used for incomplete windows at the end of the sequence.",
    )

    args = parser.parse_args()

    results = run_inference_on_mat(
        mat_path=args.mat_path,
        checkpoint_path=args.checkpoint,
        window_size=args.window_size,
        stride=args.stride,
        label_map_path=args.label_map_json,
        device_str=args.device,
        pad_value=args.pad_value,
    )

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {len(results)} window predictions to {out_path}")
    else:
        print(f"Computed {len(results)} window predictions.")
        print("Showing first 10:")
        for row in results[:10]:
            print(
                f"win={row['window_index']:3d}  "
                f"t={row['time']:7.3f}  "
                f"idx={row['pred_idx']:2d}  "
                f"label={row['pred_label']}  "
                f"prob={row['prob']:.3f}"
            )


if __name__ == "__main__":
    main()
