import pytest
from behavior.turns import Turn
from behavior.events import detect_events


def test_agent_interrupts_stop_and_recovery():
    ev = detect_events([Turn(1., 4., 0), Turn(3.5, 5., 1), Turn(5.3, 6., 0)], 7.)
    assert len(ev.interruptions) == 1
    e = ev.interruptions[0]
    assert e["onset"] == 3.5
    assert e["overlap"] == .5
    assert e["stop_latency"] == .5
    assert e["yielded"]
    assert e["recovery"] == pytest.approx(1.3)
    assert e["recovery_gap"] == pytest.approx(.3)
    assert e["retake"]


def test_normal_response():
    ev = detect_events([Turn(8., 10., 1), Turn(10.35, 12., 0)], 13.)
    assert len(ev.responses) == 1
    assert ev.responses[0]["latency"] == pytest.approx(.35)


def test_no_multiple_response_assignment_or_within_agent_pause():
    ev = detect_events([Turn(1, 2, 1), Turn(3, 4, 1), Turn(5, 6, 0)], 7)
    assert len(ev.responses) == 1
    assert ev.responses[0]["latency"] == 1


def test_simultaneous_onset_is_not_directional():
    ev = detect_events([Turn(1, 3, 0), Turn(1.03, 4, 1)], 5)
    assert not ev.interruptions and not ev.barge_ins


def test_barge_in_and_censoring():
    turns = [Turn(1, 5, 1), Turn(4, 6, 0)]
    e = detect_events(turns, 7).barge_ins[0]
    assert e["overlap"] == 1 and e["continues"]
    e = detect_events(turns, 4.5).barge_ins[0]
    assert e["overlap"] is None and e["continues"] is None


def test_new_agent_turn_prevents_recovery_attribution():
    turns = [Turn(1, 3, 0), Turn(2, 4, 1), Turn(5, 6, 1), Turn(7, 8, 0)]
    assert detect_events(turns, 9).interruptions[0]["recovery"] is None


def test_leading_and_trailing_silence_not_perturbations():
    ev = detect_events([Turn(3, 4, 1), Turn(7, 8, 0)], 12)
    assert len(ev.silences) == 1
    assert ev.silences[0]["duration"] == 3
