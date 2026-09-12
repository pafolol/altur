"""Versioned, label-independent definitions used in BOTH training and inference."""

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
FEATURE_VERSION = "1.0.0"


@dataclass(frozen=True)
class EventConfig:
    merge_gap_s: float = 0.15
    onset_tolerance_s: float = 0.064
    min_overlap_s: float = 0.096
    immediate_yield_s: float = 0.5
    immediate_retake_s: float = 0.5
    long_gap_s: float = 2.0
    short_agent_turn_s: float = 3.0
    cv_min_mean_s: float = 0.1
    min_variability_events: int = 3


@dataclass(frozen=True)
class VADConfig:
    threshold: float = 0.5
    hysteresis: float = 0.15
    min_speech_s: float = 0.16
    min_silence_s: float = 0.16
    pad_s: float = 0.032
    merge_gap_s: float = 0.15

    def __post_init__(self):
        if not 0 < self.threshold < 1 or not 0 <= self.hysteresis < self.threshold:
            raise ValueError("Invalid VAD probability thresholds")
        if min(self.min_speech_s, self.min_silence_s, self.pad_s, self.merge_gap_s) < 0:
            raise ValueError("VAD durations cannot be negative")


def feature_config():
    return {"version": FEATURE_VERSION, "events": asdict(EventConfig())}
