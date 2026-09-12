"""Offline Silero 6.2.1 ONNX at native 8 kHz. No torch, diarization or API calls.

The 256-sample/32-context/state protocol follows Silero's MIT OnnxWrapper.
Both channels form an independent batch of size two; state is LOCAL to each call.
Postprocessing is our tested configurable hysteresis implementation, not an exact
copy of the upstream timestamp defaults. See THIRD_PARTY_NOTICES.md.
"""

import hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
from .audio import Audio
from .config import ROOT, VADConfig
from .turns import Turn, normalize_turns, intersection_duration

VAD_RELEASE = "v6.2.1"
VAD_COMMIT = "7e30209a3e901f9842f81b225f3e93d8199902b1"
MODEL_PATH = ROOT / "artifacts" / "vad" / "silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


def probabilities_to_turns(probs, duration, config=VADConfig()):
    """Hysteresis + minimum speech/silence, pad, merge; 32 ms resolution."""
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[0] != 2 or not np.isfinite(probs).all():
        raise ValueError("Expected finite [2, frames] VAD probabilities")
    turns = []
    for ch in (0, 1):
        start = pending_end = None
        intervals = []
        for i, p in enumerate(probs[ch]):
            t = i * .032
            if t >= duration:
                break
            if p >= config.threshold:
                pending_end = None
                if start is None:
                    start = t
            elif start is not None and p < config.threshold - config.hysteresis:
                if pending_end is None:
                    pending_end = t
                if t - pending_end + 1e-9 >= config.min_silence_s:
                    if pending_end - start >= config.min_speech_s - 1e-9:
                        intervals.append((start, pending_end))
                    start = pending_end = None
        if start is not None and duration - start >= config.min_speech_s:
            # End remains censored if not enough silence has been observed.
            intervals.append((start, duration))
        for a, b in intervals:
            turns.append(Turn(max(0., a - config.pad_s), min(duration, b + config.pad_s), ch))
    return normalize_turns(turns, duration, config.merge_gap_s)


def speech_iou(a, b, channel):
    x, y = ([t for t in ts if t.channel == channel] for ts in (a, b))
    intersection = intersection_duration(x, y)
    union = sum(t.duration for t in x + y) - intersection
    return intersection / union if union else 1.


def segmentation_stability(probs, duration, config):
    """Label-free threshold sensitivity; only a quality heuristic, not a feature."""
    from dataclasses import replace
    low = probabilities_to_turns(probs, duration, replace(config, threshold=max(.2, config.threshold - .1)))
    high = probabilities_to_turns(probs, duration, replace(config, threshold=min(.9, config.threshold + .1)))
    return float(np.mean([speech_iou(low, high, ch) for ch in (0, 1)]))


class SileroVAD:
    def __init__(self, model_path=MODEL_PATH, config=VADConfig()):
        ort.disable_telemetry_events()
        self.config = config
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError("Offline VAD model missing. Run python -m scripts.download_vad during setup.")
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha256 != MODEL_SHA256:
            raise ValueError("VAD artifact checksum mismatch")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def probabilities(self, audio: Audio):
        if audio.sample_rate != 8000 or audio.samples.ndim != 2 or audio.samples.shape[1] != 2:
            raise ValueError("VAD requires separated stereo 8kHz audio")
        state = np.zeros((2, 2, 128), dtype=np.float32)
        context = np.zeros((2, 32), dtype=np.float32)
        n = len(audio.samples)
        result = np.empty((2, (n + 255) // 256), dtype=np.float32)
        for i, offset in enumerate(range(0, n, 256)):
            chunk = audio.samples[offset:offset + 256].T
            if chunk.shape[1] < 256:
                chunk = np.pad(chunk, ((0, 0), (0, 256 - chunk.shape[1])))
            x = np.concatenate([context, chunk], axis=1).astype(np.float32, copy=False)
            out, state = self.session.run(None, {"input": x, "state": state, "sr": np.array(8000, dtype=np.int64)})
            result[:, i] = out.reshape(-1)
            context = x[:, -32:]
        return result

    def extract(self, audio):
        probs = self.probabilities(audio)
        turns = probabilities_to_turns(probs, audio.duration, self.config)
        stability = segmentation_stability(probs, audio.duration, self.config)
        return turns, {"stability": stability, "frame_s": .032, "vad_version": VAD_RELEASE}
