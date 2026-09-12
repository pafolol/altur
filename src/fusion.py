"""
FUSION - the modular decision layer.

One call in, one verdict out, but the verdict is a weighted vote of independent detection systems:

    stereo 8 kHz WAV bytes
      -> layer 1  acoustic    (frozen Wav2Vec2 Spanish + MLP on the caller's voice)      -> p1, quality, abstain?
      -> layer 2  behaviour   (Silero turn timing + logistic on the interaction)         -> p2, quality, abstain?
      -> layer N  ...                                                                     -> pN, ...
      -> combine(weights, mode, abstention policy)                                        -> fused probability
      -> threshold                                                                        -> {"is_synthetic", "confidence"}

Everything that varies is data, not code:

    LAYERS          the registry. Adding a semantic (or any other) layer is ONE Layer subclass plus one
                    line in build_layers(); the endpoints, the frontend and the batch evaluation are all
                    driven by the registry and need no edit.
    FusionConfig    the tunable part: per-layer weights, how the scores are combined, what happens when a
                    layer has no evidence, and where the decision threshold sits. Default = every layer
                    weighted equally, so with the two layers of today that is exactly 50 % / 50 %.

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
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {"key": self.key, "display": self.display, "probability": round(float(self.probability), 6),
                "quality": round(float(self.quality), 4), "abstained": bool(self.abstained),
                "reason": self.reason, "latency_ms": round(float(self.latency_ms), 1), "details": self.details}


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
    """
    weights: tuple = ()               # tuple of (key, weight) pairs - hashable, so the config stays frozen
    mode: str = "weighted_mean"
    on_abstain: str = "renormalise"
    use_quality: bool = False
    threshold: float = config.DECISION_THRESHOLD
    confidence_mode: str = config.CONFIDENCE_MODE

    @classmethod
    def from_dict(cls, data, base=None):
        """Partial update: anything absent keeps the value it had in `base` (or the default)."""
        base = base or cls()
        data = data or {}
        weights = base.weight_map()
        for k, v in (data.get("weights") or {}).items():
            weights[k] = float(v)
        cfg = replace(base, weights=tuple(sorted(weights.items())),
                      mode=str(data.get("mode", base.mode)),
                      on_abstain=str(data.get("on_abstain", base.on_abstain)),
                      use_quality=bool(data.get("use_quality", base.use_quality)),
                      threshold=float(data.get("threshold", base.threshold)),
                      confidence_mode=str(data.get("confidence_mode", base.confidence_mode)))
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
        return self

    def weight_map(self):
        return dict(self.weights)

    def weight_of(self, layer):
        """A layer the config has never heard of contributes with its own default weight."""
        return self.weight_map().get(layer.key, layer.default_weight)

    def to_dict(self):
        return {"weights": self.weight_map(), "mode": self.mode, "on_abstain": self.on_abstain,
                "use_quality": self.use_quality, "threshold": self.threshold,
                "confidence_mode": self.confidence_mode}


# ============================================================================= the combiner


def _logit(p):
    p = float(np.clip(p, _EPS, 1 - _EPS))
    return float(np.log(p / (1 - p)))


def _sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -40, 40))))


def combine(results, cfg, weights_by_key=None):
    """
    The whole decision, as pure arithmetic on numbers the layers already produced.

    `results`         list of LayerResult (or of the dicts LayerResult.to_dict() makes - the frontend
                      sends those straight back when it re-tunes the weights).
    `weights_by_key`  the per-layer weights; defaults to the ones in `cfg`, falling back to the layer's
                      own default for anything the config does not mention.

    Returns a dict with the fused probability, the verdict, and - the part that matters for a demo -
    exactly how much each layer contributed to it.
    """
    rows = [r if isinstance(r, LayerResult) else LayerResult(**{k: v for k, v in r.items()
                                                               if k in LayerResult.__dataclass_fields__})
            for r in results]
    weights_by_key = dict(weights_by_key or cfg.weight_map())

    contributions, total = [], 0.0
    for r in rows:
        raw = float(weights_by_key.get(r.key, 0.0))
        # Two different reasons a layer can drop out: the user turned it off (weight 0), or the layer
        # itself reported that it had nothing to go on. They are shown separately on the page.
        if r.abstained and cfg.on_abstain == "renormalise":
            effective = 0.0
        else:
            effective = raw * (r.quality if cfg.use_quality else 1.0)
        p = 0.5 if r.abstained else float(np.clip(r.probability, 0.0, 1.0))
        contributions.append({"key": r.key, "display": r.display, "probability": p,
                              "raw_probability": float(r.probability), "quality": float(r.quality),
                              "abstained": bool(r.abstained), "reason": r.reason,
                              "weight": raw, "effective_weight": effective})
        total += effective

    if total <= 0:
        # Nobody had both evidence and weight. Never guess: sit on the fence and say why.
        fused = 0.5
        used = []
        note = ("every layer abstained" if rows and all(r.abstained for r in rows)
                else "no layer carries weight" if rows else "no layer is loaded")
    else:
        for c in contributions:
            c["share"] = c["effective_weight"] / total
        used = [c for c in contributions if c["effective_weight"] > 0]
        if cfg.mode == "logit_mean":
            fused = _sigmoid(sum(c["share"] * _logit(c["probability"]) for c in used))
        else:
            fused = sum(c["share"] * c["probability"] for c in used)
        note = ""
    for c in contributions:
        c.setdefault("share", 0.0)

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
        "layers": contributions,
        "config": cfg.to_dict() | {"weights": weights_by_key},
    }


# ============================================================================= the layers


class Layer:
    """
    One detection system.

    Implement `available()`, `load()` and `score()`; everything else - the HTTP surface, the weight
    slider, the batch evaluation, the report - works off this interface and never knows what is inside.
    """
    key = "layer"
    display = "Layer"
    description = ""
    default_weight = 1.0

    def __init__(self):
        self._loaded = False
        self.load_error = None

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

    # -- shared ---------------------------------------------------------------
    def ensure_loaded(self):
        if not self._loaded:
            self._load()
            self._loaded = True

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
                "default_weight": self.default_weight, "available": bool(self.available()),
                "loaded": self._loaded, **self.info()}


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
    display = "Acoustic V1 (specialist)"
    description = "Frozen Wav2Vec2 Spanish layer 5 + MLP on the caller's voice, Platt-calibrated."

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
                "signal": "the caller's voice (timbre, articulation, codec/vocoder traces)"}


class BehaviourLayer(Layer):
    """
    The conversation instead of the voice: separated-channel Silero VAD at 8 kHz -> turns -> interruption,
    barge-in and response-latency timing -> 24 features -> logistic + sigmoid calibration. The frozen
    module in behaviour/ is imported and used as-is; no file in it is edited.

    It reports its OWN evidence quality and abstains outright ("insufficient_evidence") on calls with
    fewer than two interaction events - a caller who never interacts leaves nothing to time.
    """
    key = "behaviour"
    display = "Behaviour (conversation timing)"
    description = "Separated-channel Silero VAD -> turn-taking timing -> 24 features -> calibrated logistic."

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
            "insufficient evidence: fewer than two interaction events" if abstained else "",
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
                "signal": "when the caller speaks, yields, interrupts and answers - not how they sound"}


# --------------------------------------------------------------------------- the registry
#
# THIS is the list to extend. A new detection system becomes a full citizen of the endpoint, the page
# and the batch evaluation by appearing here; nothing downstream hard-codes a layer key.
#
def build_layers(acoustic_model="wav2vec2_spanish", include_unavailable=False):
    layers = [
        AcousticLayer(acoustic_model, display=_acoustic_display(acoustic_model)),
        BehaviourLayer(),
        # SemanticLayer(),   <- the next one goes here
    ]
    return [l for l in layers if include_unavailable or l.available()]


def _acoustic_display(model_dir):
    return {"wav2vec2_spanish": "Acoustic V1 (specialist)",
            "robust_v2": "Acoustic V2 (phone-hardened)"}.get(model_dir, f"Acoustic ({model_dir})")


# ============================================================================= the detector


class FusionDetector:
    """Holds the layers and the current configuration; scores calls through all of them."""

    def __init__(self, layers=None, cfg=None, acoustic_model="wav2vec2_spanish"):
        self.layers = list(layers) if layers is not None else build_layers(acoustic_model)
        self.config = cfg or self.default_config()

    def default_config(self):
        """Equal weight to every available layer - with today's two layers, exactly 50 % / 50 %."""
        return FusionConfig(weights=tuple(sorted((l.key, 1.0 / max(len(self.layers), 1)) for l in self.layers)))

    def keys(self):
        return [l.key for l in self.layers]

    def describe(self):
        return [l.describe() | {"weight": self.config.weight_of(l)} for l in self.layers]

    def preload(self):
        for l in self.layers:
            l.ensure_loaded()
        return self

    def score_layers(self, wav_bytes):
        """Run every layer. This is the expensive half; the result is cacheable and re-combinable."""
        return [l.score(wav_bytes) for l in self.layers]

    def score(self, wav_bytes, cfg=None):
        results = self.score_layers(wav_bytes)
        return combine(results, cfg or self.config) | {"layer_details": [r.to_dict() for r in results]}


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
