"""Observable temporal events. These are proxies, NOT semantic/causal annotations."""

from dataclasses import dataclass, field

from .config import EventConfig
from .turns import Turn, normalize_turns, speaking_states


@dataclass
class Events:
    interruptions: list[dict] = field(default_factory=list)
    barge_ins: list[dict] = field(default_factory=list)
    responses: list[dict] = field(default_factory=list)
    silences: list[dict] = field(default_factory=list)


def detect_events(turns: list[Turn], duration: float, config: EventConfig = EventConfig()):
    """One event per initiating onset; responses pair adjacent floor transitions.

    Simultaneous onsets within tolerance are NOT directional interruptions.
    Outcomes at the recording/prefix boundary are right-censored (None).
    Recovery is the next caller onset after a yield, before a NEW agent turn.
    It is not called a recovery if another agent turn intervened.
    """
    turns = normalize_turns(turns, duration, config.merge_gap_s)
    caller = [t for t in turns if t.channel == 0]
    agent = [t for t in turns if t.channel == 1]
    ev = Events()
    for i, a in enumerate(agent):
        for c in caller:
            overlap = min(c.end, a.end) - a.start
            if c.start + config.onset_tolerance_s < a.start < c.end and overlap >= config.min_overlap_s:
                observed = c.end < duration - 1e-9
                yielded = (c.end < a.end - config.onset_tolerance_s) if observed else None
                resumed = next((n for n in caller if n.start >= c.end), None)
                next_agent = agent[i + 1].start if i + 1 < len(agent) else duration
                if resumed and resumed.start >= next_agent:
                    resumed = None
                recovery = resumed.start - c.end if yielded and resumed else None
                ev.interruptions.append({
                    "onset": a.start,
                    "overlap": overlap,
                    "stop_latency": c.end - a.start if observed else None,
                    "yielded": yielded,
                    "immediate_yield": (yielded and c.end - a.start <= config.immediate_yield_s) if observed else None,
                    "recovery": recovery,
                    "recovery_gap": resumed.start - a.end if yielded and resumed and a.end < duration else None,
                    "retake": (resumed.start <= a.end + config.immediate_retake_s) if yielded and resumed else None,
                })
    for c in caller:
        for a in agent:
            overlap = min(c.end, a.end) - c.start
            if a.start + config.onset_tolerance_s < c.start < a.end and overlap >= config.min_overlap_s:
                resolved = min(c.end, a.end) < duration - 1e-9
                ev.barge_ins.append({
                    "onset": c.start,
                    "overlap": overlap if resolved else None,
                    "lead": a.end - c.start if a.end < duration else None,
                    "continues": c.end > a.end + config.onset_tolerance_s if a.end < duration else None,
                })
    states = speaking_states(turns, duration)
    # Exactly agent-only -> silence -> caller-only (or direct boundary handoff).
    for i, (start, end, state) in enumerate(states):
        if state != 2 or i + 1 == len(states):
            continue
        j = i + 1
        if states[j][2] == 0:
            j += 1
        if j < len(states) and states[j][2] == 1:
            onset = states[j][0]
            a = next((a for a in agent if abs(a.end - end) < 1e-8), None)
            if a is not None:
                ev.responses.append({"onset": onset, "latency": onset - end, "agent_duration": a.duration})
    # Interior silence only. No claim that silence was unexpected/intentional.
    for i, (start, end, state) in enumerate(states):
        if state == 0 and start > 0 and end < duration and end - start >= config.long_gap_s:
            ev.silences.append({"onset": start, "duration": end - start, "caller_next": states[i + 1][2] == 1})
    return ev
