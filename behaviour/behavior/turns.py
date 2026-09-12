"""Half-open [start,end) speech intervals, measured in seconds from WAV start."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True, order=True)
class Turn:
    start: float
    end: float
    channel: int

    def __post_init__(self):
        if self.channel not in (0, 1):
            raise ValueError("Turn channel must be 0 (caller) or 1 (agent)")
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError("Non-finite turn timestamp")
        if not 0 <= self.start < self.end:
            raise ValueError("Turn timestamps must satisfy 0 <= start < end")

    @property
    def duration(self):
        return self.end - self.start


def load_turns(path: Path) -> list[Turn]:
    data = json.loads(Path(path).read_text())
    return [Turn(float(t["start"]), float(t["end"]), t["channel"]) for t in data["turns"]]


def normalize_turns(turns: list[Turn], duration: float, merge_gap: float = 0.15) -> list[Turn]:
    """Clip to observed audio and union same-channel intervals; never merge channels."""
    if not math.isfinite(duration) or duration <= 0 or merge_gap < 0:
        raise ValueError("Invalid duration or merge gap")
    result = []
    for channel in (0, 1):
        merged = []
        for t in sorted(t for t in turns if t.channel == channel and t.start < duration):
            clipped = Turn(t.start, min(duration, t.end), channel)
            if merged and clipped.start <= merged[-1].end + merge_gap + 1e-9:
                previous = merged.pop()
                merged.append(Turn(previous.start, max(previous.end, clipped.end), channel))
            else:
                merged.append(clipped)
        result.extend(merged)
    return sorted(result)


def intersection_duration(a: list[Turn], b: list[Turn]) -> float:
    """Linear-time intersection of two sorted, internally disjoint interval lists."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i].end, b[j].end) - max(a[i].start, b[j].start))
        if a[i].end <= b[j].end:
            i += 1
        else:
            j += 1
    return total


def speaking_states(turns: list[Turn], duration: float):
    """Return (start,end,state), state 0=silence,1=caller,2=agent,3=overlap."""
    changes = {0.0: [0, 0], duration: [0, 0]}
    for t in turns:
        changes.setdefault(t.start, [0, 0])[t.channel] += 1
        changes.setdefault(t.end, [0, 0])[t.channel] -= 1
    active = [0, 0]
    out = []
    points = sorted(changes)
    for start, end in zip(points, points[1:]):
        delta = changes[start]
        active = [active[c] + delta[c] for c in (0, 1)]
        state = int(active[0] > 0) + 2 * int(active[1] > 0)
        if out and out[-1][2] == state:
            out[-1] = (out[-1][0], end, state)
        else:
            out.append((start, end, state))
    return out
