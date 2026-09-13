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


def test_default_config_uses_each_layers_own_default_weight_and_role():
    """Shipped: acoustic 0.50 and behaviour 0.50 as primaries, semantic 0.15 as the verifier."""
    det = fusion.FusionDetector(layers=fusion.build_layers(include_unavailable=True))
    assert det.keys() == ["acoustic", "behaviour", "semantic"]
    assert det.config.weight_map() == {"acoustic": 0.50, "behaviour": 0.50, "semantic": 0.15}
    assert det.config.role_map() == {"acoustic": "primary", "behaviour": "primary",
                                     "semantic": "verifier"}
    # primaries that do not settle it -> all three vote, 0.50 / 0.50 / 0.15 normalised
    shares = {c["key"]: c["share"] for c in
              combine([L("acoustic", 0.6), L("behaviour", 0.6), L("semantic", 0.6)], det.config)["layers"]}
    assert shares == {"acoustic": pytest.approx(0.5 / 1.15), "behaviour": pytest.approx(0.5 / 1.15),
                      "semantic": pytest.approx(0.15 / 1.15)}


def test_each_layers_own_latency_survives_the_combiner():
    """
    Everything that reads a verdict rather than the raw LayerResults - the demo's scene payload, the
    live-call verdict, the call log - gets its per-layer timings from here. When combine() dropped them
    the demo page reported "0 ms" for a layer that had really taken half a second.
    """
    a, b = L("acoustic", 0.9), L("behaviour", 0.1)
    a.latency_ms, b.latency_ms = 482.37, 61.4
    by = {c["key"]: c for c in combine([a, b], cfg())["layers"]}
    assert by["acoustic"]["latency_ms"] == pytest.approx(482.4)
    assert by["behaviour"]["latency_ms"] == pytest.approx(61.4)


def test_a_layer_that_never_reported_a_latency_reads_zero_rather_than_breaking():
    """combine() also takes plain dicts (the console re-tuning cached scores), which may carry no timing."""
    out = combine([{"key": "acoustic", "display": "Acoustic", "probability": 0.9},
                   {"key": "behaviour", "display": "Behaviour", "probability": 0.1}], cfg())
    assert all(c["latency_ms"] == 0.0 for c in out["layers"])


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
    assert r["decisive"] is False and r["note"] == "todas las capas se abstuvieron"


def test_all_weights_zero_is_also_a_fence_sit_not_a_crash():
    r = combine([L("a", 0.9), L("b", 0.1)], cfg(weights={"a": 0, "b": 0}))
    assert r["synthetic_probability"] == 0.5 and r["decisive"] is False


def test_no_layers_at_all():
    r = combine([], cfg(weights={}))
    assert r["synthetic_probability"] == 0.5 and r["note"] == "ninguna capa está cargada"


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
        self.scored_times = 0

    def available(self):
        return True

    def _load(self):
        if self.explode:
            raise RuntimeError("weights not found")

    def _score(self, wav_bytes):
        self.scored_times += 1
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
    assert [l.default_weight for l in layers] == [0.50, 0.50, 0.15]
    assert [l.role for l in layers] == ["primary", "primary", "verifier"]
    for l in layers:
        d = l.describe()
        assert {"key", "display", "description", "default_weight", "role", "available", "loaded"} <= set(d)


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


def test_held_out_scores_are_only_used_for_dataset_calls(tmp_path):
    """
    The semantic module's model.pkl is refitted on all 353 calls, so asking the live service about a
    validation call would flatter it. An anon_id routes to its held-out score instead; live traffic has
    no anon_id and always runs the real layer.
    """
    (tmp_path / "scores_val_trainfit.csv").write_text(
        "anon_id,split,score\ncall_aaa,val,0.9\n", encoding="utf-8")
    layer = fusion.SemanticLayer(url="http://127.0.0.1:1/detect", root=tmp_path)
    r = layer.held_out_score("call_aaa")
    assert r.probability == pytest.approx(0.9) and r.abstained is False
    assert r.details["source"] == "scores_val_trainfit.csv"
    assert layer.held_out_score("call_unknown") is None

    det = fusion.FusionDetector(layers=[_Fake("acoustic", 0.2, weight=0.5), layer])
    # with an id the held-out score is free, so it is filled in even though the primary settled the call
    known = det.score_layers(b"RIFF", anon_id="call_aaa")
    assert [x.probability for x in known] == [pytest.approx(0.2), pytest.approx(0.9)]
    assert known[1].abstained is False and known[1].scored is True
    # without one, and with the primary unsure, the layer really is called - and abstains because
    # nothing is listening on port 1
    unsure = fusion.FusionDetector(layers=[_Fake("acoustic", 0.55, weight=0.5), layer])
    live = unsure.score_layers(b"RIFF")
    assert live[1].abstained is True


def test_the_train_fit_file_wins_over_the_modules_random_fold_file(tmp_path):
    """Both files may know a call; the speaker-disjoint train-only fit is the one that counts."""
    (tmp_path / "scores_semantico.csv").write_text(
        "anon_id,split,score\ncall_aaa,val,0.10\ncall_bbb,train,0.20\n", encoding="utf-8")
    (tmp_path / "scores_val_trainfit.csv").write_text(
        "anon_id,split,score\ncall_aaa,val,0.90\n", encoding="utf-8")
    layer = fusion.SemanticLayer(url="http://127.0.0.1:1/detect", root=tmp_path)
    assert layer.held_out_score("call_aaa").probability == pytest.approx(0.90)
    assert layer.held_out_score("call_aaa").details["source"] == "scores_val_trainfit.csv"
    assert layer.held_out_score("call_bbb").probability == pytest.approx(0.20)
    assert layer.held_out_score("call_bbb").details["source"] == "scores_semantico.csv"


def test_layers_without_held_out_scores_are_scored_normally():
    """Acoustic and behaviour are fitted on train alone, so the val WAV is already a held-out input."""
    for layer in (fusion.AcousticLayer(), fusion.BehaviourLayer()):
        assert layer.held_out_score("call_anything") is None


def test_semantic_layer_abstains_when_nothing_answers():
    """Nothing is listening on port 1. The layer must abstain, not raise, and not stall the verdict."""
    layer = fusion.SemanticLayer(url="http://127.0.0.1:1/detect")
    assert layer.available() is False
    r = layer.score(b"RIFF")
    assert r.abstained is True and r.probability == 0.5 and r.reason
    det = fusion.FusionDetector(layers=[_Fake("acoustic", 0.9, weight=0.5), layer])
    assert det.score(b"RIFF")["synthetic_probability"] == pytest.approx(0.9)


def test_default_registry_uses_semantic_in_process():
    layer = fusion.build_layers(include_unavailable=True)[-1]
    assert isinstance(layer, fusion.LocalSemanticLayer)
    assert layer.info()["transport"] == "in_process"


def test_remote_semantic_is_only_an_explicit_override():
    layer = fusion.build_layers(include_unavailable=True, semantic_url="http://example.test/detect")[-1]
    assert type(layer) is fusion.SemanticLayer
    assert layer.url == "http://example.test/detect"


def test_local_semantic_scores_without_http(monkeypatch):
    class Runtime:
        BUDGET_S = 1.0

        @staticmethod
        def decode(audio):
            assert audio
            return "decoded"

        @staticmethod
        def score_call(audio, deadline, stats):
            assert audio == "decoded" and deadline
            stats["n_words"] = 12
            return 0.82, "f4+f5", ""

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected HTTP call"))
    layer = fusion.LocalSemanticLayer()
    layer.runtime = Runtime
    layer._loaded = True
    result = layer.score(b"RIFF")
    assert result.probability == pytest.approx(0.82)
    assert result.details["transport"] == "in_process"
    assert result.details["n_words"] == 12


# --------------------------------------------------------------------------- the verifier gate
#
# acoustic + behaviour decide. The semantic layer is a VERIFIER: it is only consulted when those two
# did not settle the call between them, because it is the one that costs a paid API call and ~2.6 s.


def gated(**kw):
    return FusionConfig.from_dict({"weights": {"acoustic": 0.5, "behaviour": 0.5, "semantic": 0.15},
                                   "roles": {"acoustic": "primary", "behaviour": "primary",
                                             "semantic": "verifier"}, **kw})


def three(a, b, sem, **kw):
    return combine([L("acoustic", a), L("behaviour", b), L("semantic", sem)], gated(**kw))


@pytest.mark.parametrize("a,b", [(0.99, 0.95), (0.02, 0.05), (0.85, 0.78), (1.0, 1.0)])
def test_confident_primaries_settle_it_and_the_verifier_is_not_consulted(a, b):
    r = three(a, b, 0.99)
    assert r["settled_by_primaries"] is True and r["verifiers_consulted"] is False
    assert r["synthetic_probability"] == pytest.approx(r["primary_probability"])
    sem = next(c for c in r["layers"] if c["key"] == "semantic")
    assert sem["consulted"] is False and sem["share"] == 0.0
    assert "no consultada" in r["note"]


def test_the_verifier_cannot_overturn_a_settled_call():
    """A confident pair plus a verifier screaming the opposite: the verifier is never asked."""
    r = three(0.97, 0.93, 0.0)
    assert r["synthetic_probability"] == pytest.approx(0.95) and r["is_synthetic"] is True


@pytest.mark.parametrize("a,b,why", [(0.95, 0.05, "they disagree"), (0.6, 0.62, "neither is committed"),
                                     (0.79, 0.79, "just under the gate")])
def test_unsure_primaries_escalate_to_the_verifier(a, b, why):
    r = three(a, b, 0.95)
    assert r["settled_by_primaries"] is False and r["verifiers_consulted"] is True, why
    sem = next(c for c in r["layers"] if c["key"] == "semantic")
    assert sem["consulted"] is True and sem["share"] == pytest.approx(0.15 / 1.15)
    assert "consultada para desempatar" in r["note"]


def test_the_verifier_breaks_a_deadlock():
    """Primaries exactly split; the verifier is what decides the verdict."""
    for sem, verdict in ((0.95, True), (0.05, False)):
        r = three(0.95, 0.05, sem)
        assert r["primary_probability"] == pytest.approx(0.5)
        assert r["is_synthetic"] is verdict


def test_the_gate_is_tunable():
    assert three(0.85, 0.78, 0.99, verify_threshold=0.80)["verifiers_consulted"] is False
    assert three(0.85, 0.78, 0.99, verify_threshold=0.90)["verifiers_consulted"] is True
    assert three(0.60, 0.62, 0.99, verify_threshold=0.55)["verifiers_consulted"] is False


def test_use_verifiers_false_makes_it_a_plain_three_way_vote():
    """The switch turns the GATE off, not the verifier: with no gate every layer votes on every call."""
    r = three(0.97, 0.93, 0.0, use_verifiers=False)
    assert r["verifiers_consulted"] is True and r["settled_by_primaries"] is True
    sem = next(c for c in r["layers"] if c["key"] == "semantic")
    assert sem["consulted"] is True and sem["share"] == pytest.approx(0.15 / 1.15)
    assert r["synthetic_probability"] == pytest.approx((0.5 * 0.97 + 0.5 * 0.93) / 1.15)


def test_a_verifier_that_was_never_asked_cannot_vote_even_if_the_gate_reopens():
    """Re-tuning the gate after the fact must not turn an unscored layer into a number."""
    rows = [L("acoustic", 0.6), L("behaviour", 0.62),
            LayerResult("semantic", "S", 0.5, 0.0, False, "no consultada", scored=False)]
    r = combine(rows, gated())
    sem = next(c for c in r["layers"] if c["key"] == "semantic")
    assert sem["consulted"] is False and sem["share"] == 0.0
    assert r["synthetic_probability"] == pytest.approx(0.61)


def test_primaries_that_all_abstain_still_escalate():
    r = combine([L("acoustic", 0.9, abstained=True), L("behaviour", 0.1, abstained=True),
                 L("semantic", 0.9)], gated())
    assert r["settled_by_primaries"] is False
    assert r["synthetic_probability"] == pytest.approx(0.9)


def test_the_executor_does_not_even_call_a_verifier_it_does_not_need():
    """The point of the role: a settled call must not cost a paid API request."""
    sem = _Fake("semantic", 0.9, weight=0.15)
    sem.role = "verifier"
    det = fusion.FusionDetector(layers=[_Fake("acoustic", 0.97, weight=0.5),
                                        _Fake("behaviour", 0.93, weight=0.5), sem])
    r = det.score(b"x")
    assert sem.scored_times == 0
    assert r["synthetic_probability"] == pytest.approx(0.95)
    assert next(c for c in r["layers"] if c["key"] == "semantic")["scored"] is False


def test_the_executor_does_call_the_verifier_when_the_primaries_are_split():
    sem = _Fake("semantic", 0.9, weight=0.15)
    sem.role = "verifier"
    det = fusion.FusionDetector(layers=[_Fake("acoustic", 0.95, weight=0.5),
                                        _Fake("behaviour", 0.05, weight=0.5), sem])
    r = det.score(b"x")
    assert sem.scored_times == 1
    assert r["synthetic_probability"] > 0.5


def test_equal_config_turns_the_gate_off_so_an_even_split_is_really_even():
    det = fusion.FusionDetector(layers=fusion.build_layers(include_unavailable=True))
    cfg = det.equal_config()
    assert cfg.use_verifiers is False and set(cfg.role_map().values()) == {"primary"}
    r = combine([L(k, 0.9) for k in det.keys()], cfg)
    assert all(c["share"] == pytest.approx(1 / 3) for c in r["layers"])


@pytest.mark.parametrize("bad", [{"verify_threshold": 0.2}, {"verify_threshold": 1.5},
                                 {"roles": {"acoustic": "referee"}}])
def test_invalid_role_configuration_is_rejected(bad):
    with pytest.raises(ValueError):
        FusionConfig.from_dict(bad)
