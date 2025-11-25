from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any


class EventType(str, Enum):
    """
    Umpire signal event types for ViSig cricket data.

    Values are the same strings used as class labels in training and in
    your label_map_json, e.g.:

        boundary4, boundary6, cancelcall, deadball,
        legbye, noball, out, penaltyrun, shortrun, wide
    """

    BOUNDARY4 = "boundary4"
    BOUNDARY6 = "boundary6"
    CANCELCALL = "cancelcall"
    DEADBALL = "deadball"
    LEGBYE = "legbye"
    NOBALL = "noball"
    OUT = "out"
    PENALTYRUN = "penaltyrun"
    SHORTRUN = "shortrun"
    WIDE = "wide"


@dataclass
class UmpireEvent:
    """
    A discrete, post-processed umpire event detected from window-level predictions.

    Attributes
    ----------
    timestamp : float
        Event time (seconds), typically the average timestamp of the windows
        that contributed to this event.
    event_type : EventType
        Which umpire signal was detected.
    confidence : float
        Aggregate confidence for this event (e.g., mean or max of per-window
        probabilities in the run that produced it).
    """

    timestamp: float
    event_type: EventType
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a JSON-serializable dict with your preferred field names.
        """
        d = asdict(self)
        # Serialize Enum as its value (e.g. "boundary6")
        d["event_type"] = self.event_type.value
        # Rename timestamp key to a shorter "t" if you like
        d["t"] = d.pop("timestamp")
        return d
