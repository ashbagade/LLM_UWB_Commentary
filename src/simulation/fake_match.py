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


def generate_batch_commentary(
    llm_client: LLMClient,
    match_log: List[Dict[str, Any]],
) -> Dict[int, str]:
    """
    Ask Gemini ONCE to generate commentary for all balls.

    match_log entries must contain:
      - ball_index
      - event_type
      - confidence
      - score_before {runs, wickets, balls_bowled, overs}
      - score_after  {runs, wickets, balls_bowled, overs}

    Returns:
      dict mapping ball_index -> commentary string
    """
    if not match_log:
        return {}

    # Build ball-by-ball description to feed the LLM.
    lines: List[str] = []
    for rec in match_log:
        b = rec["ball_index"]
        ev = rec["event_type"]
        conf = rec["confidence"]
        sb = rec["score_before"]
        sa = rec["score_after"]
        lines.append(
            f"Ball {b}:\n"
            f"- event_type: {ev}\n"
            f"- confidence: {conf:.2f}\n"
            f"- score_before: {sb['runs']}/{sb['wickets']} "
            f"after {sb['balls_bowled']} balls ({sb['overs']} overs)\n"
            f"- score_after:  {sa['runs']}/{sa['wickets']} "
            f"after {sa['balls_bowled']} balls ({sa['overs']} overs)\n"
        )

    # More lively, broadcast-style instructions with an example.
    instructions = (
        "You are a lively TV cricket commentator calling a T20 innings. "
        "You love using vivid language, talking about momentum, pressure, and how the innings is unfolding. "
        "You must stay consistent with the ball-by-ball data: do NOT invent impossible details like player names, "
        "venue names, or scores that do not match the numbers given.\n\n"
        "For each ball, write 1–3 sentences of broadcast-style commentary as if you are speaking live on air. "
        "Commentary for ball i should take into account only balls 1..i (earlier context is fine, but never future balls).\n\n"
        "Return your answer as a strict JSON array with this structure:\n"
        "[\n"
        '  {"ball_index": 1, "commentary": "SIX! Rohit launches the very first ball into the stands, '
        'sending a clear message that the bowlers will be under pressure from the start."},\n'
        '  {"ball_index": 2, "commentary": "Much tighter from the bowler this time, just a nudged single '
        'into the leg side as the batsmen rotate the strike and settle in."}\n'
        "]\n"
        "Do not include any text before or after the JSON.\n"
    )

    content = instructions + "\nBall-by-ball data:\n\n" + "\n".join(lines)
    messages = [{"role": "user", "content": content}]

    # Rough upper bound: ~120 tokens per ball, capped at 2048.
    max_tokens = min(2048, 120 * len(match_log))

    raw = llm_client.generate(messages, max_tokens=max_tokens, temperature=0.9)
    raw = raw.strip()

    # Strip to the JSON array if there is extra text (defensive).
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        json_str = raw[start : end + 1]
    else:
        json_str = raw

    try:
        data = json.loads(json_str)
    except Exception as e:
        print(f"[WARN] Failed to parse JSON from LLM output: {e}")
        return {}

    commentary_map: Dict[int, str] = {}
    if isinstance(data, list):
        for item in data:
            try:
                idx = int(item.get("ball_index"))
                cmt = str(item.get("commentary", "")).strip()
                if cmt:
                    commentary_map[idx] = cmt
            except Exception:
                continue

    return commentary_map

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
    history_limit: int = 30,  # unused in batch mode, kept for CLI compatibility
    seed: Optional[int] = None,
    log_json: Optional[str] = None,
) -> None:
    """
    Build a pseudo-match by sampling .mat files and then calling Gemini ONCE
    to get commentary for all balls.
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

    state = MatchState()
    match_log: List[Dict[str, Any]] = []
    current_time = 0.0
    ball_index = 0

    print(f"Building pseudo-match with {len(selected_paths)} balls (batch LLM)...\n")

    # 1) Detection + state update for all balls (no LLM yet)
    for mat_path in selected_paths:
        ball_index += 1

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
            print(
                f"[Ball {ball_index}] {mat_path.name}: "
                "no confident events detected, skipping."
            )
            continue

        # Pick the best event from this trial
        best = max(events, key=lambda e: (e.confidence, -e.timestamp))

        # Synthetic global match time
        current_time += time_step_sec
        event_for_match = UmpireEvent(
            timestamp=current_time,
            event_type=best.event_type,
            confidence=best.confidence,
        )

        # Snapshot score BEFORE applying this ball
        score_before = {
            "runs": state.total_runs,
            "wickets": state.wickets,
            "balls_bowled": state.balls_bowled,
            "overs": _format_overs(state.balls_bowled),
        }

        # Update state
        apply_event_to_state(state, event_for_match)

        # Snapshot score AFTER this ball
        score_after = {
            "runs": state.total_runs,
            "wickets": state.wickets,
            "balls_bowled": state.balls_bowled,
            "overs": _format_overs(state.balls_bowled),
        }

        match_log.append(
            {
                "ball_index": ball_index,
                "file": mat_path.name,
                "event_type": event_for_match.event_type.value,
                "confidence": event_for_match.confidence,
                "match_time": event_for_match.timestamp,
                "score_before": score_before,
                "score_after": score_after,
            }
        )

    print("\nFinal match state after detection (before commentary):")
    print(
        f"  Score: {state.total_runs}/{state.wickets} "
        f"after {state.balls_bowled} balls ({_format_overs(state.balls_bowled)} overs)"
    )
    print(f"  Fours: {state.fours}, Sixes: {state.sixes}")

    if not match_log:
        print("\nNo balls with detected events; nothing to send to LLM.")
        return

    # 2) Single batched Gemini call for all commentary
    llm_client = LLMClient(system_prompt=DEFAULT_SYSTEM_PROMPT, model_name=model_name)
    commentary_map = generate_batch_commentary(llm_client, match_log)

    print("\n=== Ball-by-ball commentary ===\n")
    for rec in match_log:
        b = rec["ball_index"]
        ev = rec["event_type"]
        cmt = commentary_map.get(b, "(no commentary)")
        sb = rec["score_before"]
        sa = rec["score_after"]

        print(f"Ball {b}: file={rec['file']}")
        print(f"  Detected event: {ev} (conf={rec['confidence']:.2f})")
        print(
            f"  Score before ball: {sb['runs']}/{sb['wickets']} "
            f"after {sb['balls_bowled']} balls ({sb['overs']} overs)"
        )
        print(
            f"  Score after ball:  {sa['runs']}/{sa['wickets']} "
            f"after {sa['balls_bowled']} balls ({sa['overs']} overs)"
        )
        print("  Commentary:")
        for line in cmt.splitlines():
            print(f"    {line}")
        print("-" * 70)

        # Attach commentary back into log
        rec["commentary"] = cmt

    # 3) Optional JSON log
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
            "detecting umpire events, updating match state, and generating LLM commentary "
            "in a single batched Gemini call."
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
        help="(Unused in batch mode; kept for compatibility).",
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
