"""Explicit Behaviour-only adapter policy; no fusion weights or score recalibration."""


DECISION_POLICY_VERSION = "1.1-low-evidence-abstention"


def evidence_state(result):
    """Zero quality / fewer than two events is abstention, never a positive vote."""
    if result["quality_score"] <= 0 or result["event_count"] < 2:
        return "insufficient_evidence"
    return "available"


def challenge_response(result, threshold):
    """Resolve the Boolean-only demo contract without calling missing evidence synthetic.

    False with confidence 0.5 is a NON-FLAG fallback, not evidence of a human.
    Team fusion must consume /behavior (including quality), not this Boolean.
    Positive-quality predictions retain the frozen trained threshold and probability.
    """
    if evidence_state(result) == "insufficient_evidence":
        return {"is_synthetic": False, "confidence": 0.5}
    p = result["synthetic_probability"]
    positive = p >= threshold
    return {"is_synthetic": positive, "confidence": p if positive else 1 - p}
