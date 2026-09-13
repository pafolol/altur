"""
FUSION - the modular decision layer.

One call in, one verdict out, but the verdict is a weighted vote of independent detection systems:

    stereo 8 kHz WAV bytes
      -> PRIMARY  acoustic    (frozen Wav2Vec2 Spanish + MLP on the caller's voice)      -> p, quality, abstain?   w 0.50
      -> PRIMARY  behaviour   (Silero turn timing + logistic on the interaction)         -> p, quality, abstain?   w 0.50
      -> combine(weights, mode, abstention policy)                                        -> p_primary
         |
         |-- confident enough (>= verify_threshold, 0.80)?  -> that IS the verdict; stop here
         |
         `-- unsure? consult the VERIFIERS and re-combine:
             -> VERIFIER semantic  (Scribe + Gemini rubric, paid, ~2.6 s)                -> p, quality, abstain?   w 0.15
      -> threshold                                                                        -> {"is_synthetic", "confidence"}

Everything that varies is data, not code:

    LAYERS          the registry. Adding a semantic (or any other) layer is ONE Layer subclass plus one
                    line in build_layers(); the endpoints, the frontend and the batch evaluation are all
                    driven by the registry and need no edit.
    FusionConfig    the tunable part: per-layer weights and ROLES, how the scores are combined, what
                    happens when a layer has no evidence, where the decision threshold sits, and how
                    confident the primaries must be before a verifier is skipped. The shipped setting is
                    acoustic 0.50 / behaviour 0.50 as primaries, semantic 0.15 as a verifier at a 0.80
                    confidence gate. Every one of them is live-tunable. A layer that is missing or has no
                    evidence leaves its share to the others.

The split between `Layer.score()` (expensive, runs the models) and `combine()` (pure arithmetic on
numbers already computed) is deliberate: the frontend scores a call once and then re-tunes the weights
as often as it likes without any model ever running again.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
import config

BEHAVIOUR_ROOT = config.PROJECT_ROOT / "behaviour"

COMBINE_MODES = ("weighted_mean", "logit_mean")
ABSTAIN_POLICIES = ("renormalise", "neutral")
ROLES = ("primary", "verifier")
_EPS = 1e-6


# ============================================================================= results


@dataclass
class LayerResult:
    """What one detection system says about one call."""
    key: str
    display: str
    probability: float = 0.5          # P(synthetic) in [0, 1]
    quality: float = 0.0              # how much evidence this layer actually had, in [0, 1]
    abstained: bool = False           # True = the layer saw no usable evidence; its probability means nothing
    reason: str = ""                  # why it abstained / failed, for the page to show
    latency_ms: float = 0.0
    scored: bool = True               # False = never even asked (a verifier the primaries made unnecessary)
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {"key": self.key, "display": self.display, "probability": round(float(self.probability), 6),
                "quality": round(float(self.quality), 4), "abstained": bool(self.abstained),
                "reason": self.reason, "latency_ms": round(float(self.latency_ms), 1),
                "scored": bool(self.scored), "details": self.details}


@dataclass(frozen=True)
class FusionConfig:
    """
    The knobs the frontend exposes.

    weights         {layer key: non-negative number}. Only the RATIO matters - they are normalised at
                    combine time - so a 0..1 slider per layer behaves the way a user expects and any
                    number of layers works. A layer missing from the dict falls back to its own default.
    mode            "weighted_mean"  average the probabilities        (what "50 % / 50 %" literally means)
                    "logit_mean"     average the log-odds instead     (sharper; two confident layers that
                                                                       agree reinforce each other)
    on_abstain      "renormalise"    a layer with no evidence is dropped and the rest share its weight
                    "neutral"        it stays in at p = 0.5, pulling the fused score toward the middle
    use_quality     multiply each weight by that layer's evidence quality (off by default: with it off,
                    50 / 50 means exactly 50 / 50 on every call)
    threshold       decision threshold on the fused probability
    confidence_mode "verdict"    confidence = P(the returned verdict is right) = max(p, 1 - p)
                    "synthetic"  confidence = P(synthetic) = the fused score itself

    roles           {layer key: "primary" | "verifier"}. Primaries always vote. A verifier is consulted
                    ONLY when the primaries did not settle the call on their own. Anything the map does
                    not mention keeps the layer's own `role`.
    verify_threshold  the confidence the primaries have to reach for a verifier to be skipped. At 0.80,
                    a call the acoustic and behaviour layers jointly call at p >= 0.80 or p <= 0.20 never
                    reaches the semantic service.
    use_verifiers   False turns the second stage off entirely: every layer becomes a plain weighted vote.
    """
    weights: tuple = ()               # tuple of (key, weight) pairs - hashable, so the config stays frozen
    roles: tuple = ()                 # tuple of (key, role) pairs, same reason
    mode: str = "weighted_mean"
    on_abstain: str = "renormalise"
    use_quality: bool = False
    threshold: float = config.DECISION_THRESHOLD
    confidence_mode: str = config.CONFIDENCE_MODE
    verify_threshold: float = 0.80
    use_verifiers: bool = True

    @classmethod
    def from_dict(cls, data, base=None):
        """Partial update: anything absent keeps the value it had in `base` (or the default)."""
        base = base or cls()
        data = data or {}
        weights = base.weight_map()
        for k, v in (data.get("weights") or {}).items():
            weights[k] = float(v)
        roles = base.role_map()
        for k, v in (data.get("roles") or {}).items():
            roles[k] = str(v)
        cfg = replace(base, weights=tuple(sorted(weights.items())),
                      roles=tuple(sorted(roles.items())),
                      mode=str(data.get("mode", base.mode)),
                      on_abstain=str(data.get("on_abstain", base.on_abstain)),
                      use_quality=bool(data.get("use_quality", base.use_quality)),
                      threshold=float(data.get("threshold", base.threshold)),
                      confidence_mode=str(data.get("confidence_mode", base.confidence_mode)),
                      verify_threshold=float(data.get("verify_threshold", base.verify_threshold)),
                      use_verifiers=bool(data.get("use_verifiers", base.use_verifiers)))
        cfg.validate()
        return cfg

    def validate(self):
        if self.mode not in COMBINE_MODES:
            raise ValueError(f"mode must be one of {COMBINE_MODES}, got {self.mode!r}")
        if self.on_abstain not in ABSTAIN_POLICIES:
            raise ValueError(f"on_abstain must be one of {ABSTAIN_POLICIES}, got {self.on_abstain!r}")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be strictly between 0 and 1, got {self.threshold}")
        if self.confidence_mode not in ("verdict", "synthetic"):
            raise ValueError(f"confidence_mode must be 'verdict' or 'synthetic', got {self.confidence_mode!r}")
        for k, w in self.weights:
            if not np.isfinite(w) or w < 0:
                raise ValueError(f"weight for {k!r} must be finite and >= 0, got {w}")
        for k, r in self.roles:
            if r not in ROLES:
                raise ValueError(f"role for {k!r} must be one of {ROLES}, got {r!r}")
        if not 0.5 <= self.verify_threshold <= 1.0:
            raise ValueError(f"verify_threshold is a confidence, so it must be in [0.5, 1.0], "
                             f"got {self.verify_threshold}")
        return self

    def weight_map(self):
        return dict(self.weights)

    def role_map(self):
        return dict(self.roles)

    def weight_of(self, layer):
        """A layer the config has never heard of contributes with its own default weight."""
        return self.weight_map().get(layer.key, layer.default_weight)

    def weight_of_key(self, key, default=0.0):
        return self.weight_map().get(key, default)

    def role_of(self, layer):
        return self.role_map().get(layer.key, layer.role)

    def to_dict(self):
        return {"weights": self.weight_map(), "roles": self.role_map(), "mode": self.mode,
                "on_abstain": self.on_abstain, "use_quality": self.use_quality,
                "threshold": self.threshold, "confidence_mode": self.confidence_mode,
                "verify_threshold": self.verify_threshold, "use_verifiers": self.use_verifiers}


# ============================================================================= the combiner


def _logit(p):
    p = float(np.clip(p, _EPS, 1 - _EPS))
    return float(np.log(p / (1 - p)))


def _sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -40, 40))))


def _mix(contributions, mode):
    """Weighted mean (of probabilities, or of log-odds) over whatever is left carrying weight."""
    total = sum(c["effective_weight"] for c in contributions)
    if total <= 0:
        return 0.5, [], 0.0
    for c in contributions:
        c["share"] = c["effective_weight"] / total
    used = [c for c in contributions if c["effective_weight"] > 0]
    if mode == "logit_mean":
        return _sigmoid(sum(c["share"] * _logit(c["probability"]) for c in used)), used, total
    return sum(c["share"] * c["probability"] for c in used), used, total


def combine(results, cfg, weights_by_key=None):
    """
    The whole decision, as pure arithmetic on numbers the layers already produced.

    `results`         list of LayerResult (or of the dicts LayerResult.to_dict() makes - the frontend
                      sends those straight back when it re-tunes the weights).
    `weights_by_key`  the per-layer weights; defaults to the ones in `cfg`, falling back to the layer's
                      own default for anything the config does not mention.

    TWO STAGES, because not every layer costs the same thing to ask.

      1. The PRIMARY layers decide - acoustic and behaviour, 50 / 50. Both run offline on this machine,
         so they always run.
      2. If that decision is already confident (|p - 0.5| puts it at or above cfg.verify_threshold, 0.80
         by default) it stands, and the VERIFIER layers are not consulted at all. Only when the primaries
         are unsure - they disagree, or neither is committed - is a verifier mixed in to break the tie.

    The semantic layer is the verifier: it is the one that costs a paid ASR call, a paid LLM call and
    ~2.6 s of network, so spending it on calls the other two already agree about is waste. This is a
    decision policy, not a shortcut - the result says for every call whether it was consulted and why.

    Returns a dict with the fused probability, the verdict, and exactly how much each layer contributed.
    """
    rows = [r if isinstance(r, LayerResult) else LayerResult(**{k: v for k, v in r.items()
                                                               if k in LayerResult.__dataclass_fields__})
            for r in results]
    weights_by_key = dict(weights_by_key or cfg.weight_map())
    roles = cfg.role_map()

    contributions = []
    for r in rows:
        raw = float(weights_by_key.get(r.key, 0.0))
        # Two different reasons a layer can drop out: the user turned it off (weight 0), or the layer
        # itself reported that it had nothing to go on. They are shown separately on the page.
        if r.abstained and cfg.on_abstain == "renormalise":
            effective = 0.0
        else:
            effective = raw * (r.quality if cfg.use_quality else 1.0)
        p = 0.5 if r.abstained else float(np.clip(r.probability, 0.0, 1.0))
        if not r.scored:
            effective = 0.0        # never asked: it cannot vote, whatever the weight says
        contributions.append({"key": r.key, "display": r.display, "probability": p,
                              "raw_probability": float(r.probability), "quality": float(r.quality),
                              "abstained": bool(r.abstained), "reason": r.reason,
                              "role": roles.get(r.key, "primary"), "scored": bool(r.scored),
                              "weight": raw, "effective_weight": effective, "share": 0.0,
                              "consulted": bool(r.scored),
                              # How long THIS layer took. It is the layer's own measurement and it has to
                              # survive combine(), or every consumer that reads the verdict rather than the
                              # raw LayerResults - the scene payload, the live-call verdict, the call log -
                              # has to show a 0 it cannot stand behind.
                              "latency_ms": round(float(r.latency_ms or 0.0), 1)})

    primaries = [c for c in contributions if c["role"] == "primary"]
    verifiers = [c for c in contributions if c["role"] == "verifier"]

    # ---- stage 1: the primaries, on their own
    p_primary, used_primary, _ = _mix([dict(c) for c in primaries], cfg.mode)
    primary_confidence = max(p_primary, 1 - p_primary) if used_primary else 0.0
    settled = bool(used_primary) and primary_confidence >= cfg.verify_threshold

    # ---- stage 2: consult the verifiers only if the primaries did not settle it
    # use_verifiers=False switches the SECOND STAGE off, not the verifiers: with no gate, every layer is
    # a plain weighted vote. The gate only ever removes a verifier from a call the primaries settled.
    gate_on = bool(verifiers) and cfg.use_verifiers
    consult = bool(verifiers) and (not gate_on or not settled)
    for c in verifiers:
        # A verifier votes only if the gate opened AND it was actually asked. The executor applies the
        # same gate before spending the call, so on live traffic these two agree; on the evaluation path
        # every verifier is scored up front (a held-out lookup is free) so the gate can be re-tuned after.
        if not consult or not c["scored"]:
            c["effective_weight"] = 0.0
            c["consulted"] = False
    voting = primaries + (verifiers if consult else [])
    for c in contributions:
        if c not in voting:
            c["effective_weight"] = 0.0
    fused, used, total = _mix(contributions, cfg.mode)

    if total <= 0:
        fused, used = 0.5, []
        note = ("todas las capas se abstuvieron" if rows and all(r.abstained for r in rows)
                else "ninguna capa tiene peso" if rows else "ninguna capa está cargada")
    elif not gate_on:
        note = ""
    elif settled:
        note = (f"las primarias la resolvieron con {primary_confidence:.0%} de confianza "
                f"(>= {cfg.verify_threshold:.0%}); "
                + ", ".join(c["display"] for c in verifiers) + " no consultada")
    else:
        note = (f"las primarias solo alcanzaron {primary_confidence:.0%} de confianza "
                f"(< {cfg.verify_threshold:.0%}); "
                + ", ".join(c["display"] for c in verifiers) + " consultada para desempatar")

    if used:
        is_synthetic = bool(fused >= cfg.threshold)
        confidence = max(fused, 1 - fused) if cfg.confidence_mode == "verdict" else fused
    else:
        # Nothing voted. p = 0.5 would clear a threshold of 0.5 and call the caller synthetic on no
        # evidence at all, so the fence-sit is resolved explicitly: do not flag, and say the confidence
        # is nil. This is the behaviour module's own non-flag policy - False here is NOT a vote for human.
        is_synthetic, confidence = False, 0.5
    return {
        "is_synthetic": is_synthetic,
        "confidence": round(float(confidence), 4),
        "synthetic_probability": round(float(fused), 6),
        "decisive": bool(used),
        "note": note,
        "n_layers_used": len(used),
        "primary_probability": round(float(p_primary), 6),
        "primary_confidence": round(float(primary_confidence), 4),
        "verifiers_consulted": bool(consult and used),
        "settled_by_primaries": settled,
        "layers": contributions,
        "config": cfg.to_dict() | {"weights": weights_by_key},
    }


# ============================================================================= the layers


class Layer:
    """
    One detection system.

    Implement `available()`, `load()` and `score()`; everything else - the HTTP surface, the weight
    slider, the batch evaluation, the report - works off this interface and never knows what is inside.

    `default_weight` is this layer's share of the vote before anyone touches a fader. The shipped
    numbers are acoustic 0.50 / behaviour 0.35 / semantic 0.15; they are normalised at combine time, so
    a layer that is missing or abstaining simply leaves its share to the others.
    """
    key = "layer"
    display = "Layer"
    description = ""
    default_weight = 1.0
    role = "primary"        # "primary" votes on every call; "verifier" only when the primaries are unsure

    # A layer whose load just failed is not retried on every single call: a batch of 71 would otherwise
    # pay one connection timeout each. It abstains instantly until the window passes, then tries again.
    RETRY_AFTER_S = 30.0

    def __init__(self):
        self._loaded = False
        self.load_error = None
        self._retry_after = 0.0

    # -- to implement ---------------------------------------------------------
    def available(self):
        """Cheap check: are the weights/artifacts on disk? No model is loaded here."""
        raise NotImplementedError

    def _load(self):
        """Load the model once. Called lazily on the first score()."""
        raise NotImplementedError

    def _score(self, wav_bytes):
        """WAV bytes -> LayerResult. May assume _load() has run."""
        raise NotImplementedError

    def info(self):
        """What the page shows about this layer before any call is scored."""
        return {}

    def held_out_score(self, anon_id):
        """
        A properly held-out score for one call of the official dataset, or None.

        This exists because "score the validation WAV through the deployed model" is only an honest
        measurement while the deployed model has never seen that call. It is true of the acoustic and
        behaviour layers, whose classifiers are fitted on the train split alone - they return None here
        and are simply scored like any other audio. It is NOT true of the semantic layer, whose shipped
        model.pkl is refitted on all 353 calls: asking it about a validation call would flatter it, so it
        supplies the out-of-fold score its own training produced instead.

        Only the evaluation paths (the console's 71 calls, evaluate_fusion.py) pass an anon_id. Live
        traffic never has one, so /detect always runs the real layer.
        """
        return None

    # -- shared ---------------------------------------------------------------
    def ensure_loaded(self):
        import time
        if self._loaded:
            return
        if time.monotonic() < self._retry_after:
            raise RuntimeError(self.load_error or "layer unavailable")
        try:
            self._load()
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._retry_after = time.monotonic() + self.RETRY_AFTER_S
            raise
        self._loaded = True
        self.load_error = None

    def score(self, wav_bytes):
        """Never raises: a layer that breaks abstains, so one broken layer cannot take the verdict down."""
        import time
        t0 = time.perf_counter()
        try:
            self.ensure_loaded()
            result = self._score(wav_bytes)
        except Exception as exc:
            result = LayerResult(self.key, self.display, 0.5, 0.0, True, f"{type(exc).__name__}: {exc}")
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def describe(self):
        return {"key": self.key, "display": self.display, "description": self.description,
                "default_weight": self.default_weight, "role": self.role,
                "available": bool(self.available()), "loaded": self._loaded,
                "load_error": self.load_error, **self.info()}


class AcousticLayer(Layer):
    """
    The voice itself: a frozen Wav2Vec2 Spanish encoder truncated after layer 5, mean-pooled over each
    4 s chunk of caller speech, an MLP on top, Platt-calibrated. src/predict.py is unchanged - this only
    wraps it - so the model here is bit-identical to the one the acoustic report measured.

    `model_dir` picks WHICH acoustic model answers:
        wav2vec2_spanish  V1, the specialist - trained on the Altur recordings, energy VAD
        robust_v2         V2, the same backbone with a classifier that also saw telephone-channel audio
    """
    key = "acoustic"
    display = "Acústica V1 (especialista)"
    description = "Wav2Vec2 Spanish congelado, capa 5 + MLP sobre la voz del llamante, calibrado con Platt."
    default_weight = 0.50
    role = "primary"

    def __init__(self, model_dir="wav2vec2_spanish", classifier="mlp", display=None):
        super().__init__()
        self.model_dir = model_dir
        self.classifier = classifier
        if display:
            self.display = display
        self.detector = None

    def available(self):
        return (config.MODELS_DIR / self.model_dir / "meta.json").exists()

    def _load(self):
        from predict import AcousticDetector
        self.detector = AcousticDetector(self.model_dir, self.classifier, verbose=False)

    def _score(self, wav_bytes):
        r = self.detector.predict_wav(wav_bytes)
        no_speech = r["n_chunks"] == 0
        # The evidence this layer runs on is seconds of caller speech it could actually segment. Twelve
        # seconds (three full chunks) is where the call-level mean of the chunk scores stops moving much.
        quality = 0.0 if no_speech else float(min(1.0, r["speech_s"] / 12.0))
        return LayerResult(
            self.key, self.display, float(r["synthetic_probability"]), quality, no_speech,
            r.get("warning", "") if no_speech else "",
            details={"raw_score": round(float(r["score"]), 3), "n_chunks": r["n_chunks"],
                     "n_speech_regions": r["n_speech_regions"], "speech_s": round(float(r["speech_s"]), 1),
                     "duration_s": round(float(r["duration_s"]), 1), "vad": r["vad"],
                     "channels": int(r.get("channels", 2)), "sample_rate": int(r.get("sample_rate", 8000)),
                     "model": f"{r['backbone']} layer {r['layer']} + {r['classifier']}",
                     "calibration": self.detector.calibration.get("method", "none"),
                     "model_display": f"{r['display']}, hidden layer {r['layer']}, frozen + MLP",
                     "chunk_scores": r.get("chunk_scores", []), "chunk_spans": r.get("chunk_spans", []),
                     "timings_ms": {k: round(v * 1000) for k, v in r.get("timings", {}).items()}})

    def info(self):
        if not self.available():
            return {"model_dir": self.model_dir}
        import json
        meta = json.loads((config.MODELS_DIR / self.model_dir / "meta.json").read_text())
        return {"model_dir": self.model_dir, "layer": meta.get("layer"),
                "vad": meta.get("vad", config.VAD_METHOD),
                "trained_on": meta.get("train_sets", ["altur_original"]),
                "signal": "la voz del llamante (timbre, articulación, huellas de códec o vocoder)"}


class BehaviourLayer(Layer):
    """
    The conversation instead of the voice: separated-channel Silero VAD at 8 kHz -> turns -> interruption,
    barge-in and response-latency timing -> 24 features -> logistic + sigmoid calibration. The frozen
    module in behaviour/ is imported and used as-is; no file in it is edited.

    It reports its OWN evidence quality and abstains outright ("insufficient_evidence") on calls with
    fewer than two interaction events - a caller who never interacts leaves nothing to time.
    """
    key = "behaviour"
    display = "Conversación (tiempos de la llamada)"
    description = "Silero VAD por canal -> tiempos de los turnos -> 24 características -> logística calibrada."
    default_weight = 0.50
    role = "primary"

    def __init__(self, root=BEHAVIOUR_ROOT):
        super().__init__()
        self.root = Path(root)
        self.detector = None

    @property
    def model_path(self):
        return self.root / "artifacts" / "models" / "behavior.json"

    @property
    def vad_path(self):
        return self.root / "artifacts" / "vad" / "silero_vad.onnx"

    def available(self):
        return self.model_path.exists() and self.vad_path.exists()

    def _load(self):
        import sys
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from behavior import BehaviorDetector
        self.detector = BehaviorDetector(model_path=str(self.model_path), vad_model_path=str(self.vad_path))

    def _score(self, wav_bytes):
        from behavior.decision import evidence_state
        r = self.detector.predict(wav_bytes, debug=True)
        abstained = evidence_state(r) == "insufficient_evidence"
        d = r.get("diagnostics", {})
        return LayerResult(
            self.key, self.display, float(r["synthetic_probability"]), float(r["quality_score"]), abstained,
            "evidencia insuficiente: menos de dos eventos de interacción" if abstained else "",
            details={"event_count": r["event_count"], "behavior_confidence": round(float(r["behavior_confidence"]), 4),
                     "interruptions": d.get("interruption_count"), "barge_ins": d.get("barge_in_count"),
                     "responses": d.get("response_count"), "vad_stability": round(float(d.get("vad_stability", 0)), 3),
                     "calibration": d.get("calibration"), "evidence_state": d.get("evidence_state"),
                     "timings_ms": {k: round(v) for k, v in (d.get("timing_ms") or {}).items()}})

    def info(self):
        if not self.available():
            return {"root": str(self.root)}
        import json
        meta = json.loads(self.model_path.read_text())
        return {"root": str(self.root), "n_features": len(meta["feature_names"]),
                "calibration": meta["calibration"]["method"], "vad_config": meta["metadata"]["vad_config"],
                "signal": "cuándo habla, cede, interrumpe y responde el llamante, no cómo suena"}


class SemanticLayer(Layer):
    """
    What the caller SAYS, rather than how they sound or when they speak.

        agent channel -> trap bank (the scripted moments: the deliberately wrong digit, the interruption,
                         the product the caller may not have)
        caller channel -> energy VAD -> compaction -> ElevenLabs Scribe -> words with logprobs
        -> F4 text features (disfluencies, false starts, mexicanisms, ASR logprob, grounding in the
           agent's own words) + F5, a 7-dimension Gemini rubric
        -> logistic regression, Platt-calibrated -> P(synthetic)

    This class is the optional remote transport. The default registry uses LocalSemanticLayer below so
    the semantic runtime lives in the primary server process. A remote deployment still exposes exactly
    the contract this transport needs:

        POST /detect {"audio": "<base64 stereo 8 kHz wav>"}
          -> {"is_synthetic", "confidence", "score", "abstain", "reason", "used", "ms"}

    - where `score` is the calibrated P(synthetic), `abstain` says it had no usable transcript, and
    `used` says which path answered: "f4+f5" (Scribe + rubric), "f4" (text features only, the degraded
    path when the rubric times out) or "abstain".

    Pass url= to use it. If nothing answers, the layer abstains and the fusion renormalises the remaining
    weights - it never blocks a verdict.
    """
    key = "semantic"
    display = "Contenido (lo que dice el llamante)"
    description = "Transcripción de Scribe -> características de texto + rúbrica Gemini de 7 dimensiones -> logística calibrada."
    default_weight = 0.15
    role = "verifier"

    # The degraded path is worth less than the full one: the rubric is most of the signal.
    QUALITY_BY_PATH = {"f4+f5": 1.0, "f4": 0.5}

    def __init__(self, url=None, timeout=10.0, root=None):
        super().__init__()
        import os
        self.url = url or os.getenv("SEMANTIC_URL", "http://127.0.0.1:8100/detect")
        # The judge has a hard 30 s budget for the entire request. Primaries run first, so a verifier
        # must never be allowed to consume that whole budget by itself.
        self.timeout = min(float(os.getenv("SEMANTIC_TIMEOUT_S", timeout)), 10.0)
        self.root = Path(root or config.PROJECT_ROOT / "semantic")
        self._health = None
        self._oof_cache = None

    HEALTH_TIMEOUT_S = 1.5

    def available(self):
        """One cheap GET. The service is remote, so this is the only honest way to ask."""
        import json
        import urllib.error
        import urllib.request
        base = self.url.rsplit("/detect", 1)[0]
        try:
            with urllib.request.urlopen(base + "/health", timeout=self.HEALTH_TIMEOUT_S) as r:
                self._health = json.loads(r.read())
            return True
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            self._health = None
            return False

    def _load(self):
        if not self.available():
            raise RuntimeError(f"no semantic service answering at {self.url} "
                               f"(start it, or set SEMANTIC_URL)")

    def _score(self, wav_bytes):
        import base64
        import json
        import urllib.request
        body = json.dumps({"audio": base64.b64encode(wav_bytes).decode("ascii")}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        # `score` is the calibrated P(synthetic) and is what a fusion input must be. This service sets
        # confidence == score (its README is explicit that confidence is "no 'certeza'"), so either would
        # do here - but a service that followed the challenge's own reading of "confidence" would return
        # P(the verdict is right) instead, and averaging THAT would be wrong. Read `score`; only if it is
        # absent fall back to undoing the other convention.
        if "score" in out:
            p = float(out["score"])
        else:
            c = float(out.get("confidence", 0.5))
            p = c if out.get("is_synthetic") else 1.0 - c
        path = out.get("used", "")
        abstained = bool(out.get("abstain")) or path == "abstain"
        return LayerResult(
            self.key, self.display, p, 0.0 if abstained else self.QUALITY_BY_PATH.get(path, 0.5),
            abstained, out.get("reason", "") if abstained else "",
            details={"used": path, "service_ms": out.get("ms"), "url": self.url,
                     "service_is_synthetic": out.get("is_synthetic"),
                     "service_confidence": out.get("confidence")})

    # Where a held-out score for a dataset call comes from, best first.
    #   scores_val_trainfit.csv  the served configuration fitted on the TRAIN split alone and applied to
    #                            val. Written by semantic/holdout_val.py from the caches - no API calls.
    #                            Honest against the official split, which is speaker-disjoint.
    #   scores_semantico.csv     the module's own delivery. Every row is out-of-fold, but from a RANDOM
    #                            5-fold over all 353 calls, which ignores that split; model.py's own
    #                            comment calls it "slightly optimistic". In practice the two agree to
    #                            0.002 AUC, so this is a fallback rather than a correction.
    HELD_OUT_FILES = ("scores_val_trainfit.csv", "scores_semantico.csv")

    def held_out_score(self, anon_id):
        """
        A score for one official call from a model that was not fitted on it.

        The shipped `model.pkl` is refitted on all 353 calls, so putting a validation call through the
        live service would be asking a model about audio it has already seen. The module anticipates
        this - its README calls `scores_semantico.csv` "la entrega al backend para aprender los pesos de
        fusion" - and `semantic/holdout_val.py` sharpens it to a train-only fit.
        """
        row = self._oof().get(anon_id)
        if row is None:
            return None
        return LayerResult(self.key, self.display, row["score"], 1.0, False, "",
                           details={"source": row["source"], "split": row["split"],
                                    "note": "held out: model.pkl is refitted on all 353 calls, so the "
                                            "live service must not be asked about a call it was fitted on"})

    def _oof(self):
        if self._oof_cache is None:
            import csv
            self._oof_cache = {}
            # Later files do not overwrite earlier ones: the first source that knows a call wins.
            for name in self.HELD_OUT_FILES:
                path = self.root / name
                if not path.exists():
                    continue
                with open(path, newline="") as f:
                    for r in csv.DictReader(f):
                        self._oof_cache.setdefault(r["anon_id"], {
                            "score": float(r["score"]), "split": r["split"], "source": name})
        return self._oof_cache

    def info(self):
        d = {"url": self.url, "timeout_s": self.timeout, "transport": "http",
             "signal": "las palabras mismas: lo que el llamante ofrece, repite, corrige y aterriza",
             "held_out_scores": len(self._oof()), "root": str(self.root)}
        if self._health:
            d["service"] = self._health
        return d


class LocalSemanticLayer(SemanticLayer):
    """Run the semantic detector in this process; no localhost service or second backend is required."""

    def __init__(self, timeout=10.0, root=None):
        super().__init__(url="local", timeout=timeout, root=root)
        self.runtime = None

    def available(self):
        if not (self.root / "model.pkl").exists():
            return False
        try:
            from semantic import config as semantic_config
            return all(semantic_config.keys_present().values())
        except (ImportError, SyntaxError):
            return False

    def _load(self):
        if not self.available():
            raise RuntimeError("local semantic model or ELEVENLABS_API_KEY/GEMINI_API_KEY is unavailable")
        from semantic import server as semantic_server
        self.runtime = semantic_server

    def _score(self, wav_bytes):
        import base64
        import time
        t0 = time.perf_counter()
        stats = {}
        x = self.runtime.decode(base64.b64encode(wav_bytes).decode("ascii"))
        p, path, reason = self.runtime.score_call(x, t0 + self.runtime.BUDGET_S, stats)
        p = float(np.clip(p, 0.0, 1.0))
        abstained = path == "abstain"
        return LayerResult(
            self.key, self.display, p, 0.0 if abstained else self.QUALITY_BY_PATH.get(path, 0.5),
            abstained, reason if abstained else "",
            details={"used": path, "service_ms": int((time.perf_counter() - t0) * 1000),
                     "transport": "in_process", **stats})

    def info(self):
        return {"transport": "in_process", "timeout_s": self.timeout,
                "signal": "las palabras mismas: lo que el llamante ofrece, repite, corrige y aterriza",
                "held_out_scores": len(self._oof()), "root": str(self.root)}


# --------------------------------------------------------------------------- the registry
#
# THIS is the list to extend. A new detection system becomes a full citizen of the endpoint, the page
# and the batch evaluation by appearing here; nothing downstream hard-codes a layer key.
#
# The order here is the order of the columns, the faders and the report.
#
def build_layers(acoustic_model="wav2vec2_spanish", include_unavailable=False, semantic_url=None):
    semantic = SemanticLayer(semantic_url) if semantic_url else LocalSemanticLayer()
    layers = [
        AcousticLayer(acoustic_model, display=_acoustic_display(acoustic_model)),   # primary, 0.50
        BehaviourLayer(),                                                           # primary, 0.50
        semantic,                                                                   # verifier, 0.15
    ]
    return [l for l in layers if include_unavailable or l.available()]


def _acoustic_display(model_dir):
    return {"wav2vec2_spanish": "Acústica V1 (especialista)",
            "robust_v2": "Acústica V2 (endurecida para teléfono)"}.get(model_dir, f"Acústica ({model_dir})")


# ============================================================================= the detector


class FusionDetector:
    """Holds the layers and the current configuration; scores calls through all of them."""

    def __init__(self, layers=None, cfg=None, acoustic_model="wav2vec2_spanish", semantic_url=None):
        self.layers = list(layers) if layers is not None else build_layers(acoustic_model, semantic_url=semantic_url)
        self.config = cfg or self.default_config()

    def default_config(self):
        """
        Each layer's own default_weight and role: acoustic 0.50 and behaviour 0.50 as PRIMARIES, semantic
        0.15 as the VERIFIER consulted only when those two do not settle a call between them.

        Weights are NOT renormalised here. A layer that is unavailable simply never appears and combine()
        divides by the weight actually present, so nothing has to be re-tuned when a service comes back.
        """
        return FusionConfig(weights=tuple(sorted((l.key, float(l.default_weight)) for l in self.layers)),
                            roles=tuple(sorted((l.key, l.role) for l in self.layers)))

    def equal_config(self):
        """
        Every available layer weighted the same AND voting on every call - the "assume nothing" setting
        the page also offers. Turning the verifier stage off is part of it: an equal split in which one
        layer is usually skipped would not be an equal split.
        """
        n = max(len(self.layers), 1)
        return FusionConfig(weights=tuple(sorted((l.key, 1.0 / n) for l in self.layers)),
                            roles=tuple(sorted((l.key, "primary") for l in self.layers)),
                            use_verifiers=False)

    def keys(self):
        return [l.key for l in self.layers]

    def describe(self):
        return [l.describe() | {"weight": self.config.weight_of(l)} for l in self.layers]

    def preload(self):
        """
        Warm every layer before the first request. A layer that cannot load is NOT an error here: it
        records why, abstains on every call, and the others carry the verdict. Startup never fails
        because one optional layer is not answering.
        """
        for l in self.layers:
            try:
                l.ensure_loaded()
            except Exception:
                pass  # l.load_error holds the reason; describe() and /layers report it
        return self

    def _score_one(self, layer, wav_bytes, anon_id):
        """
        `anon_id` is only ever passed by the EVALUATION paths (the console's 71 held-out calls,
        evaluate_fusion.py). A layer whose deployed model was fitted on that call answers from its own
        held-out scores instead, so the table measures the layer rather than flattering it. Live traffic
        has no anon_id, so /detect always runs every layer for real.
        """
        r = layer.held_out_score(anon_id) if anon_id else None
        return r if r is not None else layer.score(wav_bytes)

    def score_layers(self, wav_bytes, anon_id=None, cfg=None):
        """
        Run the layers, in two stages, and skip what the first stage made unnecessary.

        The primaries always run - they are local and cheap. The verifiers run only if the primaries did
        NOT settle the call, and that is the whole point of the role: the semantic layer costs a paid ASR
        call, a paid LLM call and a couple of seconds of network, so a call that the acoustic and
        behaviour layers already agree on at >= 80 % confidence never reaches it. A skipped verifier comes
        back as `scored=False` with the reason, so the page can say "not asked" rather than inventing a
        number or pretending the layer had no evidence.

        The exception is a verifier that can answer for free - a held-out score for a dataset call is a
        dict lookup. Those are always filled in, so the console has a complete table and the gate can be
        moved afterwards without re-running anything.
        """
        cfg = cfg or self.config
        primaries = [l for l in self.layers if cfg.role_of(l) == "primary"]
        verifiers = [l for l in self.layers if cfg.role_of(l) == "verifier"]

        out = [self._score_one(l, wav_bytes, anon_id) for l in primaries]
        if not verifiers:
            return out

        consult = cfg.use_verifiers and not combine(out, cfg)["settled_by_primaries"]
        for l in verifiers:
            free = l.held_out_score(anon_id) if anon_id else None
            if free is not None:
                out.append(free)
            elif consult:
                out.append(self._score_one(l, wav_bytes, anon_id))
            else:
                out.append(LayerResult(
                    l.key, l.display, 0.5, 0.0, False,
                    "no consultada: las capas primarias resolvieron esta llamada por su cuenta", scored=False))
        # Keep the registry's order, so every column, fader and row lines up with /layers.
        order = {l.key: i for i, l in enumerate(self.layers)}
        return sorted(out, key=lambda r: order.get(r.key, len(order)))

    def score(self, wav_bytes, cfg=None, anon_id=None):
        cfg = cfg or self.config
        results = self.score_layers(wav_bytes, anon_id, cfg)
        return combine(results, cfg) | {"layer_details": [r.to_dict() for r in results]}


# --------------------------------------------------------------------------- module-level convenience
_FUSION = None


def get_fusion(acoustic_model="wav2vec2_spanish"):
    global _FUSION
    if _FUSION is None:
        _FUSION = FusionDetector(acoustic_model=acoustic_model)
    return _FUSION


def predict_fused(call_wav):
    """path / bytes of a stereo 8 kHz WAV -> fused P(synthetic). The mirror of predict.predict_acoustic."""
    if isinstance(call_wav, (str, Path)):
        call_wav = Path(call_wav).read_bytes()
    return get_fusion().score(bytes(call_wav))["synthetic_probability"]


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Score calls through every detection layer and fuse the votes.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--acoustic-model", default="wav2vec2_spanish", help="wav2vec2_spanish (V1) or robust_v2")
    parser.add_argument("--weight", action="append", default=[], metavar="KEY=W",
                        help="override one layer's weight, e.g. --weight acoustic=0.7 --weight behaviour=0.3")
    parser.add_argument("--mode", default="weighted_mean", choices=COMBINE_MODES)
    parser.add_argument("--on-abstain", default="renormalise", choices=ABSTAIN_POLICIES)
    parser.add_argument("--use-quality", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    det = FusionDetector(acoustic_model=args.acoustic_model)
    if not det.layers:
        raise SystemExit("no detection layer is available: check models/ and behaviour/artifacts/")
    overrides = dict(w.split("=", 1) for w in args.weight)
    det.config = FusionConfig.from_dict({"weights": {k: float(v) for k, v in overrides.items()},
                                         "mode": args.mode, "on_abstain": args.on_abstain,
                                         "use_quality": args.use_quality}, det.config)
    print(f"[fusion] layers: {', '.join(f'{l.display} (w={det.config.weight_of(l):.2f})' for l in det.layers)}")
    out = []
    for p in args.paths:
        r = det.score(Path(p).read_bytes())
        r["file"] = p
        out.append(r)
        if not args.json:
            print(f"\n{p}")
            for c in r["layers"]:
                state = "ABSTAINED" if c["abstained"] else f"p={c['probability']:.3f}"
                print(f"  {c['display']:34s} {state:16s} weight {c['weight']:.2f} -> share {c['share']:.0%}"
                      f"   quality {c['quality']:.2f}")
            print(f"  {'FUSED':34s} p={r['synthetic_probability']:.3f} -> "
                  f"{'SYNTHETIC' if r['is_synthetic'] else 'HUMAN'} (confidence {r['confidence']:.3f})"
                  + (f"   [{r['note']}]" if r["note"] else ""))
    if args.json:
        print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
