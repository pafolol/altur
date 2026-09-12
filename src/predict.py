"""
INFERENCE - the production acoustic detector.

    stereo 8 kHz WAV (path / bytes / base64)
      -> caller channel (0)  -> VAD -> chunks <= 4 s -> RMS-normalised -> 16 kHz
      -> frozen backbone, truncated after the selected layer -> mean over time
      -> classifier (MLP or logistic regression) -> chunk log-odds
      -> call-level aggregation (mean) -> calibration -> synthetic probability

    predict_acoustic(call_wav) -> float in [0, 1]   (0 = strongly human, 1 = strongly synthetic)

The preprocessing is the SAME code that produced the training embeddings (audio.caller_chunks), so the
model never sees differently prepared audio at inference.

Run:  python src/predict.py path/to/call.wav [more.wav ...] [--backbone wavlm] [--json]
      python src/predict.py --demo            (a few validation calls with known labels)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
import audio
import config

BEST_MODEL_FILE = config.REPORTS_DIR / "best_model.json"        # winner of the three-backbone benchmark
DEPLOYED_MODEL_FILE = config.REPORTS_DIR / "deployed_model.json"  # winner of the cross-domain evaluation


def default_backbone():
    """
    Which model directory answers when nobody names one.

    deployed_model.json wins if it exists: it is written by evaluate_models.py from the measured cross-domain
    matrix, and it is what makes the fusion interface here and the HTTP endpoint in server.py agree. It falls
    back to best_model.json, the winner of the original three-backbone benchmark, and then to whatever is on
    disk. To pin a model by hand, delete or edit deployed_model.json - or just pass the name.
    """
    for f, key in ((DEPLOYED_MODEL_FILE, "dir"), (BEST_MODEL_FILE, "backbone")):
        if f.exists():
            name = json.loads(f.read_text()).get(key)
            if name and (config.MODELS_DIR / name / "meta.json").exists():
                return name
    for key in config.BACKBONE_ORDER:
        if (config.MODELS_DIR / key / "meta.json").exists():
            return key
    raise SystemExit("No trained model found: run extract_embeddings.py and train_classifier.py first.")


class AcousticDetector:
    """Loads one backbone + classifier + calibration once; scores calls."""

    def __init__(self, backbone=None, classifier="mlp", device=None, verbose=True, vad=None):
        self.backbone = backbone or default_backbone()
        self.model_dir = config.MODELS_DIR / self.backbone
        self.meta = json.loads((self.model_dir / "meta.json").read_text())
        self.layer = int(self.meta["layer"])
        self.pooling = self.meta.get("pooling", "mean")
        self.aggregation = self.meta.get("aggregation", config.CHUNK_AGGREGATION)
        # Segmentation is part of the model, not of the caller: a model trained with a different voice-activity
        # detector records it in meta.json and gets it back here. Models trained before this field existed fall
        # back to config.VAD_METHOD, so the deployed specialist is bit-identical to what was benchmarked.
        self.vad = vad if vad is not None else self.meta.get("vad", config.VAD_METHOD)
        # `backbone` is really the directory under models/. For the three benchmarked backbones it is also a
        # key of config.BACKBONES; for a model like robust_v2 it is not, so the Hugging Face name and the
        # display name come from its own meta.json.
        self.hf_name = self.meta.get("hf_name") or config.BACKBONES[self.backbone]["hf_name"]
        self.display = self.meta.get("display") or config.BACKBONES.get(self.backbone, {}).get("display", self.backbone)
        self.device = device or config.get_device(verbose=False)
        self.classifier_name = classifier
        t0 = time.perf_counter()
        self._load_backbone()
        self._load_classifier()
        self.calibration = self._load_calibration()
        self.load_s = time.perf_counter() - t0
        if verbose:
            print(f"[detector] {self.display} layer {self.layer} + {classifier}, "
                  f"calibration={self.calibration.get('method', 'none')}, device={self.device.type}, "
                  f"loaded in {self.load_s:.1f}s")

    # ------------------------------------------------------------------ loading
    def _load_backbone(self):
        from transformers import AutoFeatureExtractor, AutoModel
        name = self.hf_name
        self.do_normalize = bool(AutoFeatureExtractor.from_pretrained(name).do_normalize)
        model = AutoModel.from_pretrained(name)
        # Only the layers up to the selected one are needed: drop the rest (exact for hidden_states[layer];
        # we keep layer+1 blocks so that models with a final layer norm never apply it to our layer).
        keep = min(self.layer + 1, model.config.num_hidden_layers)
        model.encoder.layers = model.encoder.layers[:keep]
        model.config.num_hidden_layers = keep
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _load_classifier(self):
        if self.classifier_name == "mlp":
            from train_classifier import MLP
            ckpt = torch.load(self.model_dir / "mlp.pt", map_location=self.device)
            self.mlp = MLP(ckpt["dim"], ckpt["hidden"], ckpt["dropout"]).to(self.device)
            self.mlp.load_state_dict(ckpt["state_dict"])
            self.mlp.eval()
        else:
            import joblib
            self.logreg = joblib.load(self.model_dir / "logreg.joblib")

    def _load_calibration(self):
        path = self.model_dir / f"calibration_{self.classifier_name}.json"
        if path.exists():
            return json.loads(path.read_text())
        return {"method": "none", "a": 1.0, "b": 0.0}

    # ------------------------------------------------------------------ scoring
    @torch.inference_mode()
    def embed(self, chunks):
        """list of 16 kHz chunks -> (n, dim) embedding of the selected layer (mean over time)."""
        out = []
        for c in chunks:
            x = torch.from_numpy(audio.normalize_for_model(c, self.do_normalize))[None].to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                h = self.model(x, output_hidden_states=True).hidden_states[self.layer].float()
            feat = h.mean(dim=1)
            if self.pooling == "meanstd":
                feat = torch.cat([feat, h.std(dim=1)], dim=1)
            out.append(feat)
        return torch.cat(out)

    def chunk_scores(self, emb):
        if self.classifier_name == "mlp":
            return self.mlp.scores(emb)
        return self.logreg.decision_function(emb.cpu().numpy())

    def calibrate(self, score):
        a, b = self.calibration.get("a", 1.0), self.calibration.get("b", 0.0)
        return float(1.0 / (1.0 + np.exp(-(a * score + b))))

    def predict_stereo(self, stereo, sr):
        """(n, ch) float32 array + sample rate -> result dict."""
        from metrics import aggregate_chunk_scores
        t = {}
        t0 = time.perf_counter()
        chunks, spans, regions = audio.caller_chunks(stereo, sr, return_regions=True, vad_method=self.vad)
        t["segmentation_s"] = time.perf_counter() - t0
        result = {"backbone": self.backbone, "display": self.display, "layer": self.layer,
                  "classifier": self.classifier_name, "vad": self.vad,
                  "duration_s": len(stereo) / sr, "sample_rate": sr, "channels": int(stereo.shape[1]),
                  "n_speech_regions": len(regions), "n_chunks": len(chunks),
                  "speech_s": float(sum(e - s for s, e in regions))}
        if not chunks:
            result.update({"score": 0.0, "synthetic_probability": 0.5, "chunk_scores": [], "warning": "no caller speech found",
                           "timings": t})
            return result
        t0 = time.perf_counter()
        emb = self.embed(chunks)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t["backbone_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = np.asarray(self.chunk_scores(emb), dtype=float)
        score = aggregate_chunk_scores(scores, self.aggregation)
        t["classifier_s"] = time.perf_counter() - t0
        t["total_s"] = sum(t.values())
        result.update({"score": float(score), "raw_probability": float(1 / (1 + np.exp(-score))),
                       "synthetic_probability": self.calibrate(score), "chunk_scores": scores.round(3).tolist(),
                       "chunk_spans": [(round(s, 2), round(e, 2)) for s, e in spans], "timings": t})
        return result

    def predict_wav(self, source):
        """source: path, bytes, or file-like object of a WAV file."""
        t0 = time.perf_counter()
        stereo, sr = audio.read_wav(source)
        res = self.predict_stereo(stereo, sr)
        res["timings"]["decode_s"] = time.perf_counter() - t0 - res["timings"].get("total_s", 0)
        return res

    def predict_base64(self, b64):
        stereo, sr = audio.decode_base64_wav(b64)
        return self.predict_stereo(stereo, sr)

    def verdict(self, result):
        """Challenge contract: {"is_synthetic": bool, "confidence": float}."""
        p = result["synthetic_probability"]
        is_synthetic = bool(p >= config.DECISION_THRESHOLD)
        confidence = max(p, 1 - p) if config.CONFIDENCE_MODE == "verdict" else p
        return {"is_synthetic": is_synthetic, "confidence": round(float(confidence), 4)}


_DETECTOR = None


def predict_acoustic(call_wav, backbone=None):
    """
    The interface the fusion system consumes:  path / bytes / base64 string of a stereo 8 kHz WAV -> float,
    0.0 = strongly human, 1.0 = strongly synthetic. The detector is loaded once and cached.
    """
    global _DETECTOR
    if _DETECTOR is None or (backbone and _DETECTOR.backbone != backbone):
        _DETECTOR = AcousticDetector(backbone, verbose=False)
    if isinstance(call_wav, (str, Path)) and Path(call_wav).exists():
        return _DETECTOR.predict_wav(call_wav)["synthetic_probability"]
    if isinstance(call_wav, (bytes, bytearray)):
        return _DETECTOR.predict_wav(bytes(call_wav))["synthetic_probability"]
    return _DETECTOR.predict_base64(call_wav)["synthetic_probability"]


def print_result(r, path):
    v = {"is_synthetic": r["synthetic_probability"] >= config.DECISION_THRESHOLD}
    print(f"\nFile:         {path}")
    print(f"Audio:        {r['duration_s']:.1f} s, {r['channels']} ch @ {r['sample_rate']} Hz -> caller speech {r['speech_s']:.1f} s "
          f"in {r['n_speech_regions']} regions -> {r['n_chunks']} chunks")
    print(f"Model:        {r.get('display', r['backbone'])} layer {r['layer']} + {r['classifier']}")
    print(f"Verdict:      {'SYNTHETIC' if v['is_synthetic'] else 'HUMAN'}   synthetic probability {r['synthetic_probability']:.3f} "
          f"(raw score {r['score']:+.2f})")
    if r["chunk_scores"]:
        print(f"Chunk scores: {r['chunk_scores']}")
    t = r["timings"]
    print(f"Latency:      {t.get('total_s', 0) * 1000:.0f} ms  (segmentation {t.get('segmentation_s', 0) * 1000:.0f} ms, "
          f"backbone {t.get('backbone_s', 0) * 1000:.0f} ms, classifier {t.get('classifier_s', 0) * 1000:.0f} ms)")
    if "warning" in r:
        print(f"Warning:      {r['warning']}")


def main():
    parser = argparse.ArgumentParser(description="Score stereo 8 kHz call WAVs: is the caller synthetic?")
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--backbone", default=None, choices=[None] + config.BACKBONE_ORDER)
    parser.add_argument("--classifier", default="mlp", choices=["mlp", "logreg"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--demo", action="store_true", help="score 6 validation calls with known labels")
    args = parser.parse_args()
    det = AcousticDetector(args.backbone, args.classifier)
    paths = [Path(p) for p in args.paths]
    truths = {}
    if args.demo:
        import dataset
        df = dataset.load_split("val")
        for y in (0, 1):
            for row in df[df["y"] == y].sample(n=3, random_state=config.SEED).itertuples():
                paths.append(Path(row.path))
                truths[Path(row.path)] = row.label
    if not paths:
        parser.error("give at least one WAV file (or --demo)")
    results = []
    for p in paths:
        r = det.predict_wav(p)
        r["file"] = str(p)
        if p in truths:
            r["truth"] = truths[p]
        results.append(r)
        if not args.json:
            print_result(r, p)
            if p in truths:
                ok = (r["synthetic_probability"] >= 0.5) == (truths[p] == "synthetic")
                print(f"Truth:        {truths[p].upper()}  -> {'correct' if ok else 'WRONG'}")
    if args.json:
        print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
