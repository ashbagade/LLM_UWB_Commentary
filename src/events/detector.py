from __future__ import annotations

from typing import List, Tuple, Iterable

from src.events.schema import EventType, UmpireEvent


WindowPred = Tuple[float, str, float]
# (timestamp, label, prob)


def _normalize_window_preds(window_preds: Iterable[WindowPred]) -> List[WindowPred]:
    """
    Ensure window_preds is a clean list[(time, label, prob)] sorted by time.
    """
    preds = list(window_preds)
    preds.sort(key=lambda x: x[0])  # sort by time
    return preds


def detect_events(
    window_preds: List[WindowPred],
    min_conf: float = 0.8,
    min_run_length: int = 2,
    merge_gap_sec: float = 0.5,
) -> List[UmpireEvent]:
    """
    Convert noisy window-level predictions into discrete umpire events.

    Parameters
    ----------
    window_preds : list of (time, label, prob)
        Window-level predictions. 'label' should be the human-readable
        class label (e.g., "boundary6"), not an index. This matches
        the 'pred_label' field from run_on_trial.py output.
    min_conf : float, default 0.8
        Minimum per-window probability to consider a window for event
        formation. Windows below this threshold are ignored.
    min_run_length : int, default 2
        Minimum number of consecutive windows with the same label needed
        to form an event. Shorter runs are treated as noise.
    merge_gap_sec : float, default 0.5
        If two consecutive events have the same label and their timestamps
        are within this gap, they are merged into a single event.

    Returns
    -------
    events : list[UmpireEvent]
        Cleaned-up list of detected events.
    """
    preds = _normalize_window_preds(window_preds)

    # 1) Filter windows by confidence and label
    filtered: List[WindowPred] = []
    for t, label, prob in preds:
        if label is None:
            continue
        if label == "background":
            continue
        if prob < min_conf:
            continue
        filtered.append((t, label, prob))

    if not filtered:
        return []

    # 2) Group consecutive windows with the same label into runs
    runs: List[List[WindowPred]] = []
    current_run: List[WindowPred] = []
    current_label: str | None = None

    for t, label, prob in filtered:
        if current_label is None:
            current_label = label
            current_run = [(t, label, prob)]
        elif label == current_label:
            current_run.append((t, label, prob))
        else:
            # Finish previous run
            runs.append(current_run)
            # Start new run
            current_label = label
            current_run = [(t, label, prob)]

    if current_run:
        runs.append(current_run)

    # 3) Convert runs to events (apply min_run_length)
    events: List[UmpireEvent] = []
    for run in runs:
        if len(run) < min_run_length:
            continue  # discard short runs

        times = [t for t, _, _ in run]
        probs = [p for _, _, p in run]
        label = run[0][1]  # all labels in run are the same

        # Map label string to EventType; skip unknown labels gracefully
        try:
            event_type = EventType(label)
        except ValueError:
            # Label not in EventType enum; skip
            continue

        timestamp = float(sum(times) / len(times))  # mean time
        confidence = float(sum(probs) / len(probs))  # mean prob

        events.append(UmpireEvent(timestamp=timestamp, event_type=event_type, confidence=confidence))

    if not events:
        return []

    # 4) Optionally merge nearby events of the same type
    events.sort(key=lambda e: e.timestamp)
    merged: List[UmpireEvent] = []
    for ev in events:
        if not merged:
            merged.append(ev)
            continue

        last = merged[-1]
        if ev.event_type == last.event_type and (ev.timestamp - last.timestamp) <= merge_gap_sec:
            # Merge: new timestamp = average, confidence = max
            new_t = (last.timestamp + ev.timestamp) / 2.0
            new_conf = max(last.confidence, ev.confidence)
            merged[-1] = UmpireEvent(timestamp=new_t, event_type=last.event_type, confidence=new_conf)
        else:
            merged.append(ev)

    return merged
