"""
Tests for the fusion layer (src/fusion.py).

The combiner is the one piece of arithmetic that decides every verdict, and the weight sliders on the
page drive it directly, so it is tested on its own - no model, no audio, no network. The layers are
tested through a fake Layer, which also proves the point of the registry: a new detection system only
has to implement the three methods.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fusion  # noqa: E402
from fusion import FusionConfig, LayerResult, combine  # noqa: E402


def L(key, p, quality=1.0, abstained=False):
    return LayerResult(key, key.title(), p, quality, abstained)


def cfg(**kw):
    weights = kw.pop("weights", {"acoustic": 0.5, "behaviour": 0.5})
    return FusionConfig.from_dict({"weights": weights, **kw})


# --------------------------------------------------------------------------- the 50 / 50 default


def test_equal_weights_average_the_two_probabilities():
    r = combine([L("acoustic", 1.0), L("behaviour", 0.0)], cfg())
    assert r["synthetic_probability"] == pytest.approx(0.5)
    assert [c["share"] for c in r["layers"]] == [0.5, 0.5]


def test_default_config_uses_each_layers_own_default_weight():
    """The shipped split is acoustic 0.50 / behaviour 0.35 / semantic 0.15, and it comes from the layers."""
    det = fusion.FusionDetector(layers=fusion.build_layers(include_unavailable=True))
    assert det.keys() == ["acoustic", "behaviour", "semantic"]
    assert det.config.weight_map() == {"acoustic": 0.50, "behaviour": 0.35, "semantic": 0.15}
    shares = {c["key"]: c["share"] for c in
              combine([L(k, 0.9) for k in det.keys()], det.config)["layers"]}
    assert shares == {"acoustic": pytest.approx(0.50), "behaviour": pytest.approx(0.35),
                      "semantic": pytest.approx(0.15)}


def test_a_missing_layer_leaves_its_share_to_the_others():
    """With the semantic service down the registry holds two layers; 0.50 / 0.35 becomes 59 % / 41 %."""
    det = fusion.FusionDetector(layers=[_Fake("acoustic", 0.9, weight=0.50),
                                        _Fake("behaviour", 0.1, weight=0.35)])
    assert det.config.weight_map() == {"acoustic": 0.50, "behaviour": 0.35}
    shares = {c["key"]: c["share"] for c in combine(det.score_layers(b""), det.config)["layers"]}
    assert shares["acoustic"] == pytest.approx(0.50 / 0.85)
    assert shares["behaviour"] == pytest.approx(0.35 / 0.85)


def test_equal_config_is_still_available_as_the_assume_nothing_setting():
    det = fusion.FusionDetector(layers=[_Fake("a", 0.9), _Fake("b", 0.1), _Fake("c", 0.5)])
    assert det.equal_config().weight_map() == {"a": pytest.approx(1 / 3), "b": pytest.approx(1 / 3),
                                               "c": pytest.approx(1 / 3)}
    assert combine(det.score_layers(b""), det.equal_config())["synthetic_probability"] == pytest.approx(0.5)


def test_the_three_way_split_decides_the_way_the_numbers_say():
    """acoustic 0.50 + behaviour 0.35 + semantic 0.15: the two smaller layers together can outvote the big one."""
    cfg3 = FusionConfig.from_dict({"weights": {"acoustic": 0.50, "behaviour": 0.35, "semantic": 0.15}})
    r = combine([L("acoustic", 1.0), L("behaviour", 0.0), L("semantic", 0.0)], cfg3)
    assert r["synthetic_probability"] == pytest.approx(0.50)
    r = combine([L("acoustic", 0.9), L("behaviour", 0.2), L("semantic", 0.1)], cfg3)
    assert r["synthetic_probability"] == pytest.approx(0.50 * 0.9 + 0.35 * 0.2 + 0.15 * 0.1)
    # the semantic layer abstaining hands its 15 % to the other two in proportion
    r = combine([L("acoustic", 1.0), L("behaviour", 0.0), L("semantic", 0.0, abstained=True)], cfg3)
    assert r["synthetic_probability"] == pytest.approx(0.50 / 0.85)


def test_only_the_ratio_of_the_weights_matters():
    a = combine([L("acoustic", 0.9), L("behaviour", 0.1)], cfg(weights={"acoustic": 3, "behaviour": 1}))
    b = combine([L("acoustic", 0.9), L("behaviour", 0.1)], cfg(weights={"acoustic": 0.75, "behaviour": 0.25}))
    assert a["synthetic_probability"] == pytest.approx(b["synthetic_probability"]) == pytest.approx(0.7)


def test_tuning_the_weights_moves_the_verdict_across_the_threshold():
    layers = [L("acoustic", 0.9), L("behaviour", 0.2)]
    assert combine(layers, cfg(weights={"acoustic": 0.5, "behaviour": 0.5}))["is_synthetic"] is True
    assert combine(layers, cfg(weights={"acoustic": 0.3, "behaviour": 0.7}))["is_synthetic"] is False


def test_a_zero_weight_layer_is_reported_but_does_not_vote():
    r = combine([L("acoustic", 1.0), L("behaviour", 0.0)], cfg(weights={"acoustic": 1, "behaviour": 0}))
    assert r["synthetic_probability"] == pytest.approx(1.0)
    assert r["n_layers_used"] == 1
    off = next(c for c in r["layers"] if c["key"] == "behaviour")
    assert off["share"] == 0.0 and off["abstained"] is False   # off is not the same as "had no evidence"


# --------------------------------------------------------------------------- abstention


def test_renormalise_gives_an_abstaining_layers_weight_to_the_others():
    r = combine([L("acoustic", 0.8), L("behaviour", 0.0, quality=0.0, abstained=True)], cfg())
    assert r["synthetic_probability"] == pytest.approx(0.8)
    assert r["n_layers_used"] == 1


def test_neutral_keeps_an_abstaining_layer_in_at_one_half():
    r = combine([L("acoustic", 0.8), L("behaviour", 0.0, quality=0.0, abstained=True)],
                cfg(on_abstain="neutral"))
    assert r["synthetic_probability"] == pytest.approx(0.65)
    assert next(c for c in r["layers"] if c["key"] == "behaviour")["probability"] == 0.5


def test_an_abstaining_layers_own_number_is_never_used():
    """A layer that abstains may still carry a stale probability; it must not reach the average."""
    r = combine([L("acoustic", 0.8), L("behaviour", 0.99, quality=0.0, abstained=True)], cfg(on_abstain="neutral"))
    assert r["synthetic_probability"] == pytest.approx(0.65)


def test_everything_abstaining_sits_on_the_fence_and_says_so():
    r = combine([L("a", 0.9, abstained=True), L("b", 0.1, abstained=True)], cfg(weights={"a": 1, "b": 1}))
    assert r["synthetic_probability"] == 0.5
    assert r["is_synthetic"] is False and r["confidence"] == 0.5
    assert r["decisive"] is False and r["note"] == "every layer abstained"


def test_all_weights_zero_is_also_a_fence_sit_not_a_crash():
    r = combine([L("a", 0.9), L("b", 0.1)], cfg(weights={"a": 0, "b": 0}))
    assert r["synthetic_probability"] == 0.5 and r["decisive"] is False


def test_no_layers_at_all():
    r = combine([], cfg(weights={}))
    assert r["synthetic_probability"] == 0.5 and r["note"] == "no layer is loaded"


# --------------------------------------------------------------------------- modes and options


def test_logit_mean_is_sharper_than_the_probability_mean_when_layers_agree():
    layers = [L("acoustic", 0.9), L("behaviour", 0.9)]
    assert combine(layers, cfg(mode="weighted_mean"))["synthetic_probability"] == pytest.approx(0.9)
    assert combine(layers, cfg(mode="logit_mean"))["synthetic_probability"] == pytest.approx(0.9)
    # Two layers on the SAME side reinforce each other: averaging log-odds lands further from 0.5 than
    # averaging probabilities does. That is the whole reason the mode exists.
    agree = [L("acoustic", 0.99), L("behaviour", 0.6)]
    assert (combine(agree, cfg(mode="logit_mean"))["synthetic_probability"]
            > combine(agree, cfg(mode="weighted_mean"))["synthetic_probability"])
    # ... and symmetrically on the human side.
    agree_human = [L("acoustic", 0.01), L("behaviour", 0.4)]
    assert (combine(agree_human, cfg(mode="logit_mean"))["synthetic_probability"]
            < combine(agree_human, cfg(mode="weighted_mean"))["synthetic_probability"])


def test_logit_mean_survives_a_probability_of_exactly_zero_or_one():
    r = combine([L("a", 1.0), L("b", 0.0)], cfg(weights={"a": 1, "b": 1}, mode="logit_mean"))
    assert r["synthetic_probability"] == pytest.approx(0.5)


def test_use_quality_scales_the_weights_by_the_evidence():
    layers = [L("acoustic", 1.0, quality=1.0), L("behaviour", 0.0, quality=0.25)]
    assert combine(layers, cfg())["synthetic_probability"] == pytest.approx(0.5)
    assert combine(layers, cfg(use_quality=True))["synthetic_probability"] == pytest.approx(0.8)


def test_threshold_and_confidence_mode():
    layers = [L("a", 0.6), L("b", 0.6)]
    w = {"a": 1, "b": 1}
    assert combine(layers, cfg(weights=w, threshold=0.7))["is_synthetic"] is False
    assert combine(layers, cfg(weights=w, threshold=0.5))["is_synthetic"] is True
    assert combine(layers, cfg(weights=w))["confidence"] == pytest.approx(0.6)
    assert combine([L("a", 0.2), L("b", 0.2)], cfg(weights=w))["confidence"] == pytest.approx(0.8)
    assert combine([L("a", 0.2), L("b", 0.2)], cfg(weights=w, confidence_mode="synthetic"))["confidence"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- the round trip the page makes


def test_combine_accepts_the_dicts_the_page_sends_back():
    """The page caches LayerResult.to_dict() and posts them to /fusion/recombine unchanged."""
    results = [L("acoustic", 0.8, quality=0.9), L("behaviour", 0.2, quality=0.4, abstained=False)]
    direct = combine(results, cfg())
    round_tripped = combine([r.to_dict() for r in results], cfg())
    assert round_tripped["synthetic_probability"] == pytest.approx(direct["synthetic_probability"])
    assert round_tripped["is_synthetic"] == direct["is_synthetic"]


def test_partial_config_updates_keep_everything_else():
    base = cfg(weights={"acoustic": 0.3, "behaviour": 0.7}, mode="logit_mean", use_quality=True)
    updated = FusionConfig.from_dict({"weights": {"acoustic": 0.9}}, base)
    assert updated.weight_map() == {"acoustic": 0.9, "behaviour": 0.7}
    assert updated.mode == "logit_mean" and updated.use_quality is True


@pytest.mark.parametrize("bad", [{"mode": "average"}, {"on_abstain": "ignore"}, {"threshold": 0.0},
                                 {"threshold": 1.5}, {"confidence_mode": "certainty"},
                                 {"weights": {"acoustic": -1}}])
def test_invalid_configuration_is_rejected(bad):
    with pytest.raises(ValueError):
        FusionConfig.from_dict(bad)


# --------------------------------------------------------------------------- the registry contract


class _Fake(fusion.Layer):
    """Exactly what a third detection layer would have to implement."""

    def __init__(self, key, p, quality=1.0, abstained=False, explode=False, weight=1.0):
        super().__init__()
        self.key, self.display, self.p = key, key.upper(), p
        self.quality, self.abstained, self.explode = quality, abstained, explode
        self.default_weight = weight

    def available(self):
        return True

    def _load(self):
        if self.explode:
            raise RuntimeError("weights not found")

    def _score(self, wav_bytes):
        return LayerResult(self.key, self.display, self.p, self.quality, self.abstained)


def test_a_new_layer_needs_nothing_but_the_three_methods():
    det = fusion.FusionDetector(layers=[_Fake("acoustic", 1.0), _Fake("behaviour", 0.0), _Fake("semantic", 1.0)])
    r = det.score(b"")
    assert det.keys() == ["acoustic", "behaviour", "semantic"]
    assert r["synthetic_probability"] == pytest.approx(2 / 3)
    assert {c["key"] for c in r["layers"]} == {"acoustic", "behaviour", "semantic"}
    assert all(d["key"] in det.keys() for d in r["layer_details"])


def test_a_broken_layer_abstains_instead_of_taking_the_verdict_down():
    det = fusion.FusionDetector(layers=[_Fake("acoustic", 0.9), _Fake("boom", 0.5, explode=True)])
    r = det.score(b"")
    broken = next(c for c in r["layers"] if c["key"] == "boom")
    assert broken["abstained"] is True and "RuntimeError" in broken["reason"]
    assert r["synthetic_probability"] == pytest.approx(0.9)   # the working layer still decides


def test_every_layer_reports_its_own_latency():
    det = fusion.FusionDetector(layers=[_Fake("a", 0.5)])
    assert det.score(b"")["layer_details"][0]["latency_ms"] >= 0


def test_registry_builds_the_real_layers_and_they_describe_themselves():
    layers = fusion.build_layers(include_unavailable=True)
    assert [l.key for l in layers] == ["acoustic", "behaviour", "semantic"]
    assert [l.default_weight for l in layers] == [0.50, 0.35, 0.15]
    for l in layers:
        d = l.describe()
        assert {"key", "display", "description", "default_weight", "available", "loaded"} <= set(d)


# --------------------------------------------------------------------------- the semantic layer
#
# It is reached over HTTP, so these drive it against canned service responses - no socket, no API key.


class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _semantic_result(layer, payload):
    import json
    import urllib.request
    real = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: _Resp(json.dumps(payload).encode())
    try:
        return layer._score(b"RIFF")
    finally:
        urllib.request.urlopen = real


def _semantic():
    layer = fusion.SemanticLayer(url="http://127.0.0.1:1/detect")
    layer._loaded = True
    return layer


def test_semantic_layer_reads_the_probability_not_the_confidence():
    """The service returns confidence = P(the verdict is right); a fusion input must be P(synthetic)."""
    r = _semantic_result(_semantic(), {"is_synthetic": False, "confidence": 0.93, "score": 0.07,
                                       "abstain": False, "used": "f4+f5", "ms": 812})
    assert r.probability == pytest.approx(0.07)
    assert r.abstained is False and r.quality == 1.0
    assert r.details["used"] == "f4+f5"


def test_semantic_layer_reconstructs_the_probability_if_score_is_absent():
    assert _semantic_result(_semantic(), {"is_synthetic": False, "confidence": 0.93}).probability == pytest.approx(0.07)
    assert _semantic_result(_semantic(), {"is_synthetic": True, "confidence": 0.93}).probability == pytest.approx(0.93)


def test_semantic_layer_degraded_path_is_worth_less_evidence():
    full = _semantic_result(_semantic(), {"score": 0.8, "used": "f4+f5"})
    degraded = _semantic_result(_semantic(), {"score": 0.8, "used": "f4"})
    assert full.quality == 1.0 and degraded.quality == 0.5
    assert full.probability == degraded.probability == pytest.approx(0.8)


@pytest.mark.parametrize("payload", [{"score": 0.8, "abstain": True, "reason": "no_speech"},
                                     {"score": 0.5, "used": "abstain", "reason": "deadline"}])
def test_semantic_layer_abstains_when_the_service_says_so(payload):
    r = _semantic_result(_semantic(), payload)
    assert r.abstained is True and r.quality == 0.0 and r.reason == payload["reason"]


def test_semantic_layer_abstains_when_nothing_answers():
    """Nothing is listening on port 1. The layer must abstain, not raise, and not stall the verdict."""
    layer = fusion.SemanticLayer(url="http://127.0.0.1:1/detect")
    assert layer.available() is False
    r = layer.score(b"RIFF")
    assert r.abstained is True and r.probability == 0.5 and r.reason
    det = fusion.FusionDetector(layers=[_Fake("acoustic", 0.9, weight=0.5), layer])
    assert det.score(b"RIFF")["synthetic_probability"] == pytest.approx(0.9)
