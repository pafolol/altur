"""
Audio layer: everything between "a WAV file / base64 string" and "a list of 16 kHz caller speech chunks".

    stereo 8 kHz WAV  ->  caller channel (ch 0)  ->  voice activity detection  ->  speech regions
                      ->  chunks of <= CHUNK_SECONDS  ->  resampled to 16 kHz for the backbone

Every stage (training, evaluation, the HTTP endpoint) goes through THE SAME functions here, so the
model never sees differently prepared audio at inference than it saw in training.

Why an energy-based VAD and not the official turns/<id>.json files?
  The judges' benchmark sends only the WAV. Whatever segmentation we use must be computable from the
  audio alone, so we use our own detector everywhere and only use the official turn files to CHECK it
  (inspect_dataset.py measures the agreement).

About 8 kHz -> 16 kHz:
  The backbones were pretrained on 16 kHz waveforms, so we resample the caller channel to 16 kHz.
  Resampling does NOT create information: the original signal has nothing above 4 kHz (Nyquist limit
  of 8 kHz audio) and the resampled one still has nothing above 4 kHz. It only changes the
  representation to the sample rate the model expects.
"""
import base64
import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import _bootstrap  # noqa: F401  (makes `import config` work when run from src/)
import config


# ----------------------------------------------------------------------------- decoding
def read_wav(source):
    """
    source: path, bytes or file-like -> (float32 array of shape (n_samples, n_channels), sample_rate).
    Mono files come back as (n, 1) so the caller-channel logic is identical.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    wave, sr = sf.read(source, dtype="float32", always_2d=True)
    return wave, int(sr)


def decode_base64_wav(b64_string):
    """The challenge endpoint receives the WAV file bytes encoded as base64 text."""
    raw = base64.b64decode(b64_string, validate=False)
    return read_wav(raw)


def encode_wav_base64(path):
    """Inverse of decode_base64_wav: used by the tests and the client demo."""
    return base64.b64encode(open(path, "rb").read()).decode("ascii")


def get_channel(stereo, channel):
    """Caller = channel 0, agent = channel 1. A mono file is treated as the caller."""
    if stereo.shape[1] == 1:
        return stereo[:, 0].copy()
    return stereo[:, channel].copy()


def resample(wave, orig_sr, target_sr=config.MODEL_SAMPLE_RATE):
    """Polyphase resampling (anti-imaging FIR filter). 8 kHz -> 16 kHz is an exact 2x upsampling."""
    if orig_sr == target_sr:
        return wave.astype(np.float32)
    g = gcd(orig_sr, target_sr)
    return resample_poly(wave, target_sr // g, orig_sr // g).astype(np.float32)


# ----------------------------------------------------------------------------- level statistics
def frame_db(wave, sr, frame_s=config.VAD_FRAME_S, hop_s=config.VAD_HOP_S):
    """RMS level in dB of every 20 ms frame (one value every 10 ms). Returns (times_s, db)."""
    n, hop = int(frame_s * sr), int(hop_s * sr)
    if len(wave) < n:
        wave = np.pad(wave, (0, n - len(wave)))
    n_frames = 1 + (len(wave) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(wave[idx] ** 2, axis=1))
    return hop * np.arange(n_frames) / sr, 20 * np.log10(rms + 1e-9)


def rms_db(wave):
    return float(20 * np.log10(np.sqrt(np.mean(wave ** 2)) + 1e-9))


def level_stats(wave):
    """Simple per-channel numbers used by the dataset inspection and the shortcut check."""
    return {
        "rms_db": rms_db(wave),
        "peak": float(np.abs(wave).max()) if len(wave) else 0.0,
        "clipping_fraction": float(np.mean(np.abs(wave) >= 0.99)) if len(wave) else 0.0,
        "dc_offset": float(wave.mean()) if len(wave) else 0.0,
        "zero_fraction": float(np.mean(wave == 0.0)) if len(wave) else 0.0,
    }


# ----------------------------------------------------------------------------- voice activity detection
def _runs(mask):
    """Start/end indices (end exclusive) of every run of True in a boolean array."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


_SILERO = None


def _silero_model():
    global _SILERO
    if _SILERO is None:
        from silero_vad import load_silero_vad
        _SILERO = load_silero_vad()
    return _SILERO


def detect_speech_silero(wave, sr, threshold=config.SILERO_THRESHOLD, min_silence_ms=config.SILERO_MIN_SILENCE_MS,
                         min_speech_ms=config.SILERO_MIN_SPEECH_MS, pad_ms=config.SILERO_PAD_MS):
    """
    Silero VAD (a ~2 MB neural network trained on thousands of hours of speech in many languages) -> speech regions.
    Works natively at 8 kHz and 16 kHz; runs on the CPU in ~1 % of real time. Stateful, so a fresh pass per call.
    """
    import torch
    from silero_vad import get_speech_timestamps
    if sr not in (8000, 16000):
        wave, sr = resample(wave, sr, 16000), 16000
    ts = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)), _silero_model(),
                               sampling_rate=sr, threshold=threshold, min_silence_duration_ms=min_silence_ms,
                               min_speech_duration_ms=min_speech_ms, speech_pad_ms=pad_ms, return_seconds=True)
    return merge_regions([(float(t["start"]), float(t["end"])) for t in ts])


def detect_speech(wave, sr, method=None, other=None):
    """
    Speech regions of one channel with the configured method (config.VAD_METHOD).
    `other` = the opposite channel (agent), used by the crosstalk gate of the energy detector when configured.
    """
    method = method or config.VAD_METHOD
    if method == "silero":
        try:
            return detect_speech_silero(wave, sr)
        except Exception as exc:  # silero missing or failing: never leave a call without segmentation
            print(f"[audio] silero VAD failed ({exc}); falling back to energy VAD")
    return detect_speech_energy(wave, sr, other=other, crosstalk_margin_db=config.VAD_CROSSTALK_MARGIN_DB)


def detect_speech_energy(wave, sr, threshold_db=config.VAD_THRESHOLD_DB, min_abs_db=config.VAD_MIN_ABS_DB,
                  min_speech_s=config.VAD_MIN_SPEECH_S, min_gap_s=config.VAD_MIN_GAP_S, pad_s=config.VAD_PAD_S,
                  hop_s=config.VAD_HOP_S, other=None, crosstalk_margin_db=None):
    """
    Energy-based voice activity detection -> list of (start_s, end_s) speech regions ("caller turns").

      1. level of every 10 ms frame in dB
      2. noise floor = 10th percentile of those levels (the caller is silent most of the call)
      3. speech frame = level > max(floor + threshold_db, min_abs_db)
      3b. crosstalk gate (optional): if the OTHER channel (the agent) is given, a caller frame only counts when the
          caller is not much quieter than the agent at that moment - this removes the agent's echo that leaks into
          the caller channel on some lines (frames where caller_db < agent_db - margin are dropped)
      4. close gaps shorter than min_gap_s, drop regions shorter than min_speech_s, pad by pad_s
    """
    times, db = frame_db(wave, sr, hop_s=hop_s)
    floor = np.percentile(db, 10)
    speech = db > max(floor + threshold_db, min_abs_db)
    if other is not None and crosstalk_margin_db is not None:
        _, other_db = frame_db(other, sr, hop_s=hop_s)
        n = min(len(db), len(other_db))
        speech[:n] &= db[:n] >= other_db[:n] - crosstalk_margin_db
    hop_frames = lambda s: max(1, int(round(s / hop_s)))
    # close short gaps: a run of non-speech shorter than min_gap merges its neighbours
    for a, b in _runs(~speech):
        if b - a < hop_frames(min_gap_s) and a > 0 and b < len(speech):
            speech[a:b] = True
    regions = []
    for a, b in _runs(speech):
        if b - a >= hop_frames(min_speech_s):
            start = max(0.0, times[a] - pad_s)
            end = min(len(wave) / sr, times[b - 1] + config.VAD_FRAME_S + pad_s)
            regions.append((float(start), float(end)))
    return merge_regions(regions)


def merge_regions(regions):
    """Merge overlapping/touching (start, end) intervals."""
    merged = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def regions_to_mask(regions, total_s, hop_s=config.VAD_HOP_S):
    """Boolean speech mask with one entry every hop_s seconds (for comparing two segmentations)."""
    n = int(np.ceil(total_s / hop_s))
    mask = np.zeros(n, dtype=bool)
    for start, end in regions:
        mask[int(start / hop_s):int(np.ceil(end / hop_s))] = True
    return mask


def silence_stats(regions, total_s):
    """Leading silence, trailing silence, speech seconds, speech fraction, number of regions."""
    speech = sum(e - s for s, e in regions)
    return {
        "leading_silence_s": regions[0][0] if regions else total_s,
        "trailing_silence_s": total_s - regions[-1][1] if regions else total_s,
        "speech_s": speech,
        "speech_fraction": speech / total_s if total_s else 0.0,
        "n_regions": len(regions),
        "mean_region_s": speech / len(regions) if regions else 0.0,
    }


# ----------------------------------------------------------------------------- chunking
def chunk_regions(regions, chunk_s=config.CHUNK_SECONDS, min_s=config.MIN_CHUNK_SECONDS,
                  max_chunks=config.MAX_CHUNKS_PER_CALL):
    """
    Cut every speech region into pieces of at most chunk_s seconds.
    A region shorter than min_s is dropped; a leftover shorter than min_s at the end of a region is dropped.
    Chunks never cross a region boundary, so a chunk never contains a splice of two different moments.
    """
    chunks = []
    for start, end in regions:
        length = end - start
        if length < min_s:
            continue
        n_full = int(length // chunk_s)
        for i in range(n_full):
            chunks.append((start + i * chunk_s, start + (i + 1) * chunk_s))
        rest = length - n_full * chunk_s
        if rest >= min_s:
            chunks.append((start + n_full * chunk_s, end))
    return chunks[:max_chunks]


def slice_chunk(wave, sr, start_s, end_s):
    return wave[int(round(start_s * sr)):int(round(end_s * sr))]


def caller_chunks(stereo, sr, regions=None, return_regions=False, vad_method=None):
    """
    stereo (n, ch) at sr  ->  list of 16 kHz float32 chunks of the CALLER channel (+ their (start, end)).
    `regions` lets the caller override the VAD (e.g. to test the official turn files); default = our VAD.
    `vad_method` overrides config.VAD_METHOD for this call only (None = the configured default, so the
    deployed specialist keeps behaving exactly as it was trained and benchmarked).
    """
    caller = get_channel(stereo, config.CALLER_CHANNEL)
    if regions is None:
        agent = get_channel(stereo, config.AGENT_CHANNEL) if stereo.shape[1] > 1 else None
        regions = detect_speech(caller, sr, method=vad_method, other=agent)
    spans = chunk_regions(regions)
    chunks = [level_normalize(resample(slice_chunk(caller, sr, s, e), sr)) for s, e in spans]
    if return_regions:
        return chunks, spans, regions
    return chunks, spans


def level_normalize(chunk, target_rms_db=config.TARGET_RMS_DB):
    """Scale a chunk to a fixed RMS level: removes absolute loudness as a cue (same for every backbone)."""
    rms = np.sqrt(np.mean(chunk ** 2)) + 1e-9
    return (chunk * (10 ** (target_rms_db / 20) / rms)).astype(np.float32)


def normalize_for_model(chunk, do_normalize):
    """Some backbones were pretrained on zero-mean / unit-variance waveforms (preprocessor do_normalize)."""
    if not do_normalize:
        return chunk
    return ((chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-7)).astype(np.float32)
