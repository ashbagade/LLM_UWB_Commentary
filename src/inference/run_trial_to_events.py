from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from src.inference.run_on_trial import run_inference_on_mat
from src.events.detector import detect_events
from src.events.schema import UmpireEvent


def mat_to_events(
    mat_path: str,
    checkpoint_path: str,
    window_size: int,
    stride: int,
    label_map_json: str | None,
    min_conf: float,
    min_run_length: int,
    merge_gap_sec: float,
    device: str | None = None,
    pad_value: float = 0.0,
) -> List[UmpireEvent]:
    """
    Convenience wrapper: run model on .mat and detect discrete events.
    """
    # 1) Run window-level inference (reusing Milestone 0 code)
    window_preds = run_inference_on_mat(
        mat_path=mat_path,
        checkpoint_path=checkpoint_path,
        window_size=window_size,
        stride=stride,
        label_map_path=label_map_json,
        device_str=device,
        pad_value=pad_value,
    )

    # 2) Convert window_preds dicts → (time, label, prob) tuples
    tuples = []
    for row in window_preds:
        t = float(row["time"])
        label = row["pred_label"]  # may be None; detector handles this
        prob = float(row["prob"])
        tuples.append((t, label, prob))

    # 3) Detect events
    events = detect_events(
        window_preds=tuples,
        min_conf=min_conf,
        min_run_length=min_run_length,
        merge_gap_sec=merge_gap_sec,
    )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run trained model on a single ViSig .mat trial and convert "
            "window-level predictions into discrete umpire events."
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
        help="Optional path to JSON label map (label_to_idx or idx_to_label).",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.8,
        help="Minimum per-window probability to consider for event detection.",
    )
    parser.add_argument(
        "--min-run-length",
        type=int,
        default=2,
        help="Minimum number of consecutive windows with same label to form an event.",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=0.5,
        help="Merge events of the same type if within this time gap (seconds).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override: 'cuda', 'mps', or 'cpu'.",
    )
    parser.add_argument(
        "--pad-value",
        type=float,
        default=0.0,
        help="Pad value used for incomplete windows at the end of the sequence.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to write detected events as JSON.",
    )

    args = parser.parse_args()

    events = mat_to_events(
        mat_path=args.mat_path,
        checkpoint_path=args.checkpoint,
        window_size=args.window_size,
        stride=args.stride,
        label_map_json=args.label_map_json,
        min_conf=args.min_conf,
        min_run_length=args.min_run_length,
        merge_gap_sec=args.merge_gap_sec,
        device=args.device,
        pad_value=args.pad_value,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    events_dicts: List[Dict[str, Any]] = [e.to_dict() for e in events]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(events_dicts, f, indent=2)

    print(f"Wrote {len(events)} events to {out_path}")
    if events:
        print("Events:")
        for e in events:
            print(f"  t={e.timestamp:.3f}s  type={e.event_type.value}  conf={e.confidence:.3f}")


if __name__ == "__main__":
    main()
