"""
Tests for the detection-latency timeline (src/latency.py).

The claim this module makes is strong - "this is what the model would have answered after k chunks" - so
the first test checks it against the real predictor rather than against itself. The rest are the edges:
a call that never commits, a call with one chunk, thresholds that move, and an early verdict that does
not survive the call.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
import latency  # noqa: E402

CAL = {"a": 1.0, "b": 0.0}          # identity Platt: probability = sigmoid(mean log-odds)


def spans(n, chunk=4.0, gap=0.0, start=0.0):
    out, t = [], start
    for _ in range(n):
        out.append((t, t + chunk))
        t += chunk + gap
    return out


# --------------------------------------------------------------------------- the central claim


def test_the_running_value_is_what_the_model_would_have_said_on_that_prefix():
    """
    Not a re-implementation: the same aggregate_chunk_scores() the detector uses, over the prefix.
    If this ever drifts, every number on /demo becomes a story rather than a measurement.
    """
    from metrics import aggregate_chunk_scores
    scores = [2.0, -1.0, 0.5, 3.0, -2.0]
    steps = latency.progression(scores, spans(5), CAL, "mean")
    for k, s in enumerate(steps, start=1):
        expected = latency.calibrated(aggregate_chunk_scores(scores[:k], "mean"), CAL)
        assert s["probability"] == pytest.approx(round(expected, 4))
    assert steps[-1]["probability"] == pytest.approx(
        round(latency.calibrated(aggregate_chunk_scores(scores, "mean"), CAL), 4))


def test_call_time_and_caller_speech_are_different_clocks():
    """Long agent turns make them diverge, and the difference is the whole point of the page."""
    steps = latency.progression([1.0, 1.0, 1.0], spans(3, chunk=4.0, gap=20.0), CAL)
    assert [s["caller_speech_s"] for s in steps] == [4.0, 8.0, 12.0]     # only the speech counts
    assert [s["call_time_s"] for s in steps] == [4.0, 28.0, 52.0]        # the call runs on regardless


def test_each_step_also_reports_that_chunk_on_its_own():
    steps = latency.progression([3.0, -3.0], spans(2), CAL)
    assert steps[0]["chunk_probability"] > 0.9 and steps[1]["chunk_probability"] < 0.1
    assert steps[1]["probability"] == pytest.approx(0.5, abs=1e-6)        # the running mean of +3 and -3


# --------------------------------------------------------------------------- detection


def test_first_prediction_and_first_confident_detection_are_not_the_same_thing():
    # 0.60 clears the decision threshold but not the confidence band; the third chunk carries it over
    scores = [0.4, 0.4, 6.0]
    t = latency.timeline(scores, spans(3), 60.0, CAL, hi=0.85, lo=0.15)
    m = t["metrics"]
    assert m["first_prediction_after_caller_speech_sec"] == 4.0
    assert m["first_prediction_probability"] == pytest.approx(0.5987, abs=1e-3)
    assert m["confident_detection_after_caller_speech_sec"] == 12.0
    assert m["confident_detection_probability"] >= 0.85


def test_a_call_that_never_commits_reports_none_not_zero():
    """A zero would read as "detected instantly", which is the opposite of the truth."""
    t = latency.timeline([0.1, -0.1, 0.2], spans(3), 40.0, CAL, hi=0.85, lo=0.15)
    m = t["metrics"]
    assert m["never_confident"] is True
    assert m["confident_detection_at_call_sec"] is None
    assert m["seconds_before_call_end"] is None
    assert m["percentage_of_call_elapsed_at_detection"] is None
    assert m["early"] is None
    assert m["first_prediction_at_call_sec"] is not None      # it did predict, just never committed


def test_lead_time_and_share_of_the_call():
    t = latency.timeline([4.0], [(8.0, 12.0)], 100.0, CAL, hi=0.85, lo=0.15)
    m = t["metrics"]
    assert m["confident_detection_at_call_sec"] == 12.0
    assert m["seconds_before_call_end"] == 88.0
    assert m["percentage_of_call_elapsed_at_detection"] == 12.0


def test_a_detection_after_the_recording_ends_never_reports_negative_lead():
    t = latency.timeline([4.0], [(0.0, 30.0)], 20.0, CAL)
    assert t["metrics"]["seconds_before_call_end"] == 0.0
    assert t["metrics"]["percentage_of_call_elapsed_at_detection"] == 100.0


def test_the_confident_band_is_symmetric_so_human_calls_are_detected_too():
    t = latency.timeline([-4.0, -4.0], spans(2), 60.0, CAL, hi=0.85, lo=0.15)
    m = t["metrics"]
    assert m["confident_detection_at_call_sec"] == 4.0
    assert m["early"]["verdict"] == "human"


@pytest.mark.parametrize("hi,expected_chunk", [(0.60, 1), (0.85, 2), (0.999, None)])
def test_moving_the_band_moves_the_detection(hi, expected_chunk):
    # running probabilities: 0.668, 0.864, 0.903
    scores = [0.7, 3.0, 3.0]
    m = latency.timeline(scores, spans(3), 60.0, CAL, hi=hi, lo=1 - hi)["metrics"]
    assert m["confident_detection_chunk"] == expected_chunk


def test_the_thresholds_used_are_reported_so_a_number_can_be_reproduced():
    m = latency.timeline([2.0], spans(1), 10.0, CAL, hi=0.9, lo=0.1)["metrics"]
    assert m["thresholds"] == {"decision": config.DECISION_THRESHOLD,
                               "confident_synthetic": 0.9, "confident_human": 0.1}


def test_the_defaults_come_from_config_not_from_a_literal():
    assert latency.is_confident(config.CONFIDENT_SYNTHETIC_THRESHOLD) is True
    assert latency.is_confident(config.CONFIDENT_HUMAN_THRESHOLD) is True
    midpoint = (config.CONFIDENT_HUMAN_THRESHOLD + config.CONFIDENT_SYNTHETIC_THRESHOLD) / 2
    assert latency.is_confident(midpoint) is False


# --------------------------------------------------------------------------- early vs final


def test_an_early_verdict_that_does_not_survive_the_call_is_flagged():
    """The case the page must not hide: confident 'synthetic' early, 'human' by the end."""
    scores = [4.0, -3.0, -3.0, -3.0, -3.0]
    m = latency.timeline(scores, spans(5), 80.0, CAL, hi=0.85, lo=0.15)["metrics"]
    assert m["early"]["verdict"] == "synthetic"
    assert m["final"]["verdict"] == "human"
    assert m["early_agrees_with_final"] is False


def test_an_early_verdict_that_holds_is_flagged_too():
    m = latency.timeline([4.0, 4.0, 4.0], spans(3), 60.0, CAL, hi=0.85, lo=0.15)["metrics"]
    assert m["early_agrees_with_final"] is True


# --------------------------------------------------------------------------- degenerate input


def test_no_chunks_produces_no_timeline_rather_than_a_crash():
    t = latency.timeline([], [], 30.0, CAL)
    assert t["steps"] == []
    m = t["metrics"]
    assert m["first_prediction_at_call_sec"] is None and m["never_confident"] is False
    assert m["early"] is None and m["final"] is None and m["early_agrees_with_final"] is None


def test_mismatched_scores_and_spans_are_refused_rather_than_zipped_short():
    assert latency.progression([1.0, 2.0], spans(1), CAL) == []
    assert latency.progression(None, None, CAL) == []


def test_one_chunk_is_a_valid_call():
    m = latency.timeline([4.0], spans(1), 10.0, CAL)["metrics"]
    assert m["n_chunks"] == 1
    assert m["first_prediction_at_call_sec"] == m["confident_detection_at_call_sec"] == 4.0


# --------------------------------------------------------------------------- the benchmark


def test_the_benchmark_reports_medians_per_class_and_counts_what_it_missed():
    made = [latency.timeline([4.0] * 3, spans(3), 100.0, CAL),          # synthetic, detected
            latency.timeline([-4.0] * 3, spans(3), 100.0, CAL),         # human, detected
            latency.timeline([0.05] * 3, spans(3), 100.0, CAL)]         # never commits
    b = latency.benchmark(made, ["synthetic", "human", "human"])
    assert b["all"]["n"] == 3 and b["all"]["n_detected"] == 2
    assert b["all"]["n_never_confident"] == 1
    assert b["synthetic"]["n"] == 1 and b["synthetic"]["median_caller_speech_sec"] == 4.0
    assert b["human"]["n"] == 2 and b["human"]["n_detected"] == 1
    assert b["all"]["detected_before_halfway_pct"] == 100.0


def test_the_benchmark_survives_a_split_where_nothing_was_detected():
    b = latency.benchmark([latency.timeline([0.01], spans(1), 50.0, CAL)], ["human"])
    assert b["all"] == {"n": 1, "n_detected": 0}


def test_the_benchmark_needs_no_labels():
    b = latency.benchmark([latency.timeline([4.0], spans(1), 50.0, CAL)])
    assert "synthetic" not in b and b["all"]["n_detected"] == 1
