# src/simulation/fake_match.py

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.inference.run_trial_to_events import mat_to_events
from src.state.cricket_state import MatchState, apply_event_to_state
from src.commentary.llm_client import LLMClient
from src.commentary.engine import CommentaryEngine
from src.events.schema import UmpireEvent


DEFAULT_SYSTEM_PROMPT = (
    "You are a lively, knowledgeable cricket commentator calling a live T20 match. "
    "You speak concisely in 1–2 sentences per event, reacting to the latest ball "
    "while keeping an eye on the overall state of the game. You avoid emojis."
)


def _format_overs(balls_bowled: int) -> str:
    """
    Format overs as X.Y where X is full overs and Y is balls in current over (0–5).
    """
    return f"{balls_bowled // 6}.{balls_bowled % 6}"


def simulate_match_from_folder(
    data_root: str,
    checkpoint: str,
    label_map_json: str,
    num_trials: int = 20,
    window_size: int = 64,
    stride: int = 16,
    min_conf: float = 0.8,
    min_run_length: int = 2,
    merge_gap_sec: float = 0.5,
    time_step_sec: float = 30.0,
    model_name: str = "gemini-2.0-flash",
    history_limit: int = 30,
    seed: Optional[int] = None,
    log_json: Optional[str] = None,
) -> None:
    """
    Build a pseudo-match by sampling .mat files from data_root and generating
    commentary for each detected event.

    Each .mat file is treated as one ball worth of sensor data. We:
      - detect the best umpire event from that trial,
      - map it onto a synthetic match timeline,
      - update MatchState,
      - and generate LLM commentary.

    This is purely for offline demonstration, since the dataset is not a
    continuous match stream.
    """
    data_dir = Path(data_root)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data root not found: {data_dir}")

    mat_paths = sorted(data_dir.glob("*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")

    if seed is not None:
        random.seed(seed)

    if num_trials <= 0 or num_trials > len(mat_paths):
        num_trials = len(mat_paths)

    selected_paths = random.sample(mat_paths, num_trials)

    # Initialize commentary engine and match state
    llm_client = LLMClient(system_prompt=DEFAULT_SYSTEM_PROMPT, model_name=model_name)
    engine = CommentaryEngine(
        llm_client=llm_client,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        history_limit=history_limit,
    )
    state = MatchState()

    match_log: List[Dict[str, Any]] = []
    current_time = 0.0
    ball_index = 0

    print(f"Simulating pseudo-match with {len(selected_paths)} balls...\n")

    for mat_path in selected_paths:
        ball_index += 1
        # 1) Detect events for this trial
        events = mat_to_events(
            mat_path=str(mat_path),
            checkpoint_path=checkpoint,
            window_size=window_size,
            stride=stride,
            label_map_json=label_map_json,
            min_conf=min_conf,
            min_run_length=min_run_length,
            merge_gap_sec=merge_gap_sec,
            device=None,
            pad_value=0.0,
        )

        if not events:
            print(f"[Ball {ball_index}] {mat_path.name}: no confident events detected, skipping.")
            continue

        # 2) Pick the "best" event from this trial (highest confidence, earliest time)
        best = max(events, key=lambda e: (e.confidence, -e.timestamp))

        # Map onto synthetic global match time
        current_time += time_step_sec
        event_for_match = UmpireEvent(
            timestamp=current_time,
            event_type=best.event_type,
            confidence=best.confidence,
        )

        # Snapshot score before the ball
        prev_runs = state.total_runs
        prev_wkts = state.wickets
        prev_balls = state.balls_bowled

        # 3) Update match state
        apply_event_to_state(state, event_for_match)

        # 4) Generate commentary
        commentary = engine.generate_for_event(state, event_for_match)

        # 5) Print a "broadcast" style update
        print(f"Ball {ball_index}: file={mat_path.name}")
        print(f"  Detected event: {event_for_match.event_type.value} (conf={event_for_match.confidence:.2f})")
        print(
            f"  Score before ball: {prev_runs}/{prev_wkts} "
            f"after {prev_balls} balls ({_format_overs(prev_balls)} overs)"
        )
        print(
            f"  Score after ball:  {state.total_runs}/{state.wickets} "
            f"after {state.balls_bowled} balls ({_format_overs(state.balls_bowled)} overs)"
        )
        print("  Commentary:")
        for line in commentary.splitlines():
            print(f"    {line}")
        print("-" * 70)

        # 6) Append to log
        match_log.append(
            {
                "ball_index": ball_index,
                "file": mat_path.name,
                "event_type": event_for_match.event_type.value,
                "confidence": event_for_match.confidence,
                "match_time": event_for_match.timestamp,
                "score_after": {
                    "runs": state.total_runs,
                    "wickets": state.wickets,
                    "balls_bowled": state.balls_bowled,
                },
                "commentary": commentary,
            }
        )

    print("\nFinal match state:")
    print(
        f"  Score: {state.total_runs}/{state.wickets} "
        f"after {state.balls_bowled} balls ({_format_overs(state.balls_bowled)} overs)"
    )
    print(f"  Fours: {state.fours}, Sixes: {state.sixes}")

    # Optionally save log to JSON
    if log_json is not None:
        log_path = Path(log_json)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(match_log, f, indent=2)
        print(f"\nWrote match log with {len(match_log)} balls to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a pseudo cricket match by sampling .mat files from a folder, "
            "detecting umpire events, updating match state, and generating LLM commentary."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to folder containing cricket .mat files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt) from train_seq_classifier.",
    )
    parser.add_argument(
        "--label-map-json",
        type=str,
        required=True,
        help="Path to label map JSON used for inference (cricket_label_map.json).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=20,
        help="Number of .mat files (balls) to sample for the pseudo-match.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="Window size (must match training / inference).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Stride between windows (must match training / inference).",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.8,
        help="Minimum per-window confidence for event detection.",
    )
    parser.add_argument(
        "--min-run-length",
        type=int,
        default=2,
        help="Minimum run length (windows) for event detection.",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=0.5,
        help="Merge events of same type within this time gap (seconds).",
    )
    parser.add_argument(
        "--time-step-sec",
        type=float,
        default=30.0,
        help="Synthetic time gap between balls in the pseudo-match (seconds).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model name for commentary (e.g., gemini-2.0-flash).",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=30,
        help="Max number of messages to keep in commentary history (including system).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling of .mat files.",
    )
    parser.add_argument(
        "--log-json",
        type=str,
        default=None,
        help="Optional path to write a JSON log of balls, events, states, and commentary.",
    )

    args = parser.parse_args()

    simulate_match_from_folder(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        label_map_json=args.label_map_json,
        num_trials=args.num_trials,
        window_size=args.window_size,
        stride=args.stride,
        min_conf=args.min_conf,
        min_run_length=args.min_run_length,
        merge_gap_sec=args.merge_gap_sec,
        time_step_sec=args.time_step_sec,
        model_name=args.model_name,
        history_limit=args.history_limit,
        seed=args.seed,
        log_json=args.log_json,
    )


if __name__ == "__main__":
    main()
