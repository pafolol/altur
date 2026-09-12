"""
THE TELEPHONE CHANNEL - real narrowband codecs, packet loss, drift, AGC and line noise.

WHY THIS FILE EXISTS
  The deployed detector scores 100 % on the official Altur validation split, but the caller audio it sees in
  production has been through a telephone network first, and there it degrades. That is a CHANNEL SHIFT: the
  model was fitted on one recording chain and is asked to work on another. This module builds the second chain
  explicitly so we can (a) measure how much it costs us and (b) train against it.

WHAT IS REAL HERE AND WHAT IS NOT
  Real: the codecs. G.711 mu-law/A-law are the exact ITU-T companding tables (implemented here in numpy and
  unit-tested against the reference C code in Python's audioop); G.726, GSM 06.10, AMR-NB, iLBC, Speex, Opus
  and G.722 are the actual reference encoders/decoders, through ffmpeg. Real: 20 ms RTP framing, which is what
  every telephone carrier packetises to.
  Simulated: the network transport. Packet loss, its burstiness, the receiver's loss concealment, jitter-buffer
  clock drift, the analogue front-end response and the AGC are models, not captures. They are parameterised
  from the ranges telephony equipment actually uses, but they are models, and the report says so.

WHY A DISTRIBUTION OF CHANNELS AND NOT ONE CHANNEL
  Training against a single captured channel buys robustness to that channel and nothing else - change the
  carrier and you start over. So a channel is SAMPLED per call from a family of codecs and impairments
  (domain randomisation), and the families are split in two:

      SEEN   (may be used for training)            G.711 mu-law, G.711 A-law, G.726 at 32/24/16 kbit/s
      UNSEEN (evaluation only, never trained on)   GSM 06.10, AMR-NB, iLBC, Speex, Opus narrowband, G.722

  A model trained on SEEN and evaluated on UNSEEN answers the question that matters: does it survive a
  telephone channel it has never met? Nothing in UNSEEN ever reaches a gradient.

SIGNAL CONVENTION
  Everything here is float32 mono at 8 kHz (config.SOURCE_SAMPLE_RATE), the rate the challenge audio is in and
  the rate a narrowband telephone line runs at. Output length equals input length: the algorithmic delay of the
  codecs is measured by cross-correlation and compensated, so the transformed caller channel stays aligned with
  the untouched agent channel. The measured delay is reported, not hidden.
"""
import shutil
import subprocess

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

import _bootstrap  # noqa: F401
import config

SR = config.SOURCE_SAMPLE_RATE          # 8 000 Hz
FRAME_MS = 20                           # RTP packetisation interval of every telephone carrier
FRAME_SAMPLES = SR * FRAME_MS // 1000   # 160 samples = 160 bytes of G.711

FFMPEG = shutil.which("ffmpeg") or r"C:\FFmpeg\bin\ffmpeg.exe"


# ============================================================================= G.711 (exact, no ffmpeg)
# ITU-T G.711 is a logarithmic companding table: sign bit + 3-bit segment (exponent) + 4-bit mantissa. It is
# the codec of every PSTN call on earth. Implementing it directly (instead of a subprocess per call) is exact,
# vectorised and ~1000x faster; tests/test_phone_channel.py checks both directions against audioop, which is
# the reference C implementation shipped with Python.
_BIAS = 0x84            # 132
_CLIP_U = 8159          # maximum 14-bit magnitude mu-law can represent
_SEG_UEND = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
_SEG_AEND = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def pcm16_to_ulaw(pcm):
    """int16 samples -> uint8 G.711 mu-law codes (exact ITU-T G.711)."""
    val = np.asarray(pcm, dtype=np.int32) >> 2                  # 14-bit magnitude domain
    mask = np.where(val < 0, 0x7F, 0xFF).astype(np.int32)
    val = np.minimum(np.abs(val), _CLIP_U) + (_BIAS >> 2)
    seg = np.searchsorted(_SEG_UEND, val, side="left").astype(np.int32)
    code = np.where(seg >= 8, 0x7F, (seg << 4) | ((val >> (np.minimum(seg, 7) + 1)) & 0x0F))
    return (code ^ mask).astype(np.uint8)


def ulaw_to_pcm16(code):
    """uint8 G.711 mu-law codes -> int16 samples (exact ITU-T G.711)."""
    u = (~np.asarray(code, dtype=np.uint8)).astype(np.int32)
    t = ((u & 0x0F) << 3) + _BIAS
    t = t << ((u & 0x70) >> 4)
    return np.where(u & 0x80, _BIAS - t, t - _BIAS).astype(np.int16)


def pcm16_to_alaw(pcm):
    """int16 samples -> uint8 G.711 A-law codes (exact ITU-T G.711)."""
    val = np.asarray(pcm, dtype=np.int32) >> 3                  # 13-bit magnitude domain
    mask = np.where(val >= 0, 0xD5, 0x55).astype(np.int32)
    val = np.where(val >= 0, val, -val - 1)
    seg = np.searchsorted(_SEG_AEND, val, side="left").astype(np.int32)
    s = np.minimum(seg, 7)
    mantissa = np.where(s < 2, (val >> 1) & 0x0F, (val >> s) & 0x0F)
    code = np.where(seg >= 8, 0x7F, (s << 4) | mantissa)
    return (code ^ mask).astype(np.uint8)


def alaw_to_pcm16(code):
    """uint8 G.711 A-law codes -> int16 samples (exact ITU-T G.711)."""
    a = (np.asarray(code, dtype=np.uint8) ^ 0x55).astype(np.int32)
    seg = (a & 0x70) >> 4
    t = (a & 0x0F) << 4
    t = np.where(seg == 0, t + 8, np.where(seg == 1, t + 0x108, (t + 0x108) << np.maximum(seg - 1, 0)))
    return np.where(a & 0x80, t, -t).astype(np.int16)


def float_to_pcm16(w):
    return np.round(np.clip(w, -1.0, 1.0) * 32767.0).astype(np.int16)


def pcm16_to_float(p):
    return (np.asarray(p, dtype=np.float32) / 32767.0).astype(np.float32)


# ============================================================================= ffmpeg codecs
# (encoder arguments, decoder arguments, the rate the codec natively runs at). The signal goes in as raw s16le
# and comes back as raw s16le, so the only thing that happens in between is the codec itself.
FFMPEG_CODECS = {
    "g726_32k":   (["-acodec", "g726", "-b:a", "32k", "-f", "g726"], ["-f", "g726", "-code_size", "4", "-ar", "8000", "-ac", "1"], 8000),
    "g726_24k":   (["-acodec", "g726", "-b:a", "24k", "-f", "g726"], ["-f", "g726", "-code_size", "3", "-ar", "8000", "-ac", "1"], 8000),
    "g726_16k":   (["-acodec", "g726", "-b:a", "16k", "-f", "g726"], ["-f", "g726", "-code_size", "2", "-ar", "8000", "-ac", "1"], 8000),
    "gsm_fr":     (["-acodec", "libgsm", "-f", "gsm", "-ar", "8000", "-ac", "1"], ["-f", "gsm"], 8000),
    "amrnb_12k2": (["-acodec", "libopencore_amrnb", "-b:a", "12.2k", "-f", "amr", "-ar", "8000", "-ac", "1"], ["-f", "amr"], 8000),
    "amrnb_7k4":  (["-acodec", "libopencore_amrnb", "-b:a", "7.4k", "-f", "amr", "-ar", "8000", "-ac", "1"], ["-f", "amr"], 8000),
    "amrnb_4k75": (["-acodec", "libopencore_amrnb", "-b:a", "4.75k", "-f", "amr", "-ar", "8000", "-ac", "1"], ["-f", "amr"], 8000),
    "ilbc":       (["-acodec", "libilbc", "-f", "ilbc", "-ar", "8000", "-ac", "1"], ["-f", "ilbc"], 8000),
    "speex":      (["-acodec", "libspeex", "-f", "ogg", "-ar", "8000", "-ac", "1"], ["-f", "ogg"], 8000),
    "opus_nb":    (["-acodec", "libopus", "-b:a", "16k", "-application", "voip", "-f", "ogg", "-ar", "16000", "-ac", "1"], ["-f", "ogg"], 16000),
    "g722":       (["-acodec", "g722", "-f", "g722", "-ar", "16000", "-ac", "1"], ["-f", "g722"], 16000),
}


def _ffmpeg(args, data):
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", *args],
                       input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f"ffmpeg failed ({p.returncode}): {p.stderr.decode('utf8', 'ignore')[-400:]}")
    return p.stdout


def codec_roundtrip(wave, name):
    """
    8 kHz float32 -> the real encoder -> the real decoder -> 8 kHz float32.

    Codecs whose native rate is 16 kHz (Opus, G.722) are fed 16 kHz and resampled back afterwards, which is
    exactly what a gateway does when a wideband leg meets a narrowband one. The algorithmic delay some codecs
    introduce is NOT removed here; align_to() measures and removes it once, at the end of apply_channel.
    """
    if name == "g711_ulaw":
        return pcm16_to_float(ulaw_to_pcm16(pcm16_to_ulaw(float_to_pcm16(wave))))
    if name == "g711_alaw":
        return pcm16_to_float(alaw_to_pcm16(pcm16_to_alaw(float_to_pcm16(wave))))
    enc, dec, rate = FFMPEG_CODECS[name]
    src = wave if rate == SR else resample_poly(wave, rate // 1000, SR // 1000).astype(np.float32)
    packed = _ffmpeg(["-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0", *enc, "pipe:1"],
                     float_to_pcm16(src).tobytes())
    back = _ffmpeg([*dec, "-i", "pipe:0", "-f", "s16le", "-ar", str(rate), "-ac", "1", "pipe:1"], packed)
    out = pcm16_to_float(np.frombuffer(back, dtype="<i2"))
    if rate != SR:
        out = resample_poly(out, SR // 1000, rate // 1000).astype(np.float32)
    return out.astype(np.float32)


# ============================================================================= impairments
def band_limit(wave, high_pass_hz, low_pass_hz, order=6):
    """The analogue front end of a telephone line: nothing below ~300 Hz, nothing above ~3.4 kHz."""
    nyq = SR / 2
    lo = max(high_pass_hz, 20.0) / nyq
    hi = min(low_pass_hz, nyq * 0.99) / nyq
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, wave).astype(np.float32)


def spectral_tilt(wave, db_per_octave):
    """A different handset or microphone response: a constant dB-per-octave slope across the band."""
    if abs(db_per_octave) < 1e-6:
        return np.asarray(wave, dtype=np.float32)
    spec = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(len(wave), 1 / SR)
    gain = np.ones_like(freqs)
    gain[1:] = 10 ** (db_per_octave * np.log2(freqs[1:] / 1000.0) / 20)
    return np.fft.irfft(spec * gain, n=len(wave)).astype(np.float32)


def agc(wave, target_db=-20.0, window_s=0.4, max_gain_db=20.0, attack=0.15, release=0.02):
    """
    The automatic gain control every telephone gateway runs: an envelope follower pushing the level towards a
    target, fast to pull loud speech down, slow to push quiet speech up, then a hard limiter.

    This stage matters more than it looks. The Altur synthetic callers are ~6 dB louder than the human ones
    (inspect_dataset.py found it, and a classifier on trivial statistics alone reaches AUC 1.000). An AGC
    erases that difference, which is one concrete reason a model fitted on the raw dataset can fall over on a
    real line: the cue it leaned on is not there any more.
    """
    n = max(1, int(window_s * SR))
    power = np.convolve(np.asarray(wave, dtype=np.float64) ** 2, np.ones(n) / n, mode="same")
    want = np.clip(target_db - 10 * np.log10(power + 1e-12), -max_gain_db, max_gain_db)
    smoothed = np.empty_like(want)
    g = want[0]
    for i, w in enumerate(want):                       # one-pole follower, different constants up and down
        g += (attack if w < g else release) * (w - g)
        smoothed[i] = g
    return np.clip(wave * (10 ** (smoothed / 20)), -1.0, 1.0).astype(np.float32)


def gilbert_elliott(n_frames, loss_rate, mean_burst, rng):
    """
    Packet loss on a real network is bursty, not independent: the standard two-state Gilbert-Elliott model.
    q = probability of leaving the loss state (so the mean burst length is 1/q), p = probability of entering
    it; the stationary loss probability p/(p+q) is solved here for the requested loss_rate.
    """
    if loss_rate <= 0:
        return np.zeros(n_frames, dtype=bool)
    q = 1.0 / max(mean_burst, 1.0)
    p = q * loss_rate / max(1e-9, 1.0 - loss_rate)
    lost = np.zeros(n_frames, dtype=bool)
    state = False
    for i in range(n_frames):
        state = (rng.random() >= q) if state else (rng.random() < p)
        lost[i] = state
    return lost


def packet_loss(wave, loss_rate, mean_burst, concealment, rng):
    """
    Cut the signal into 20 ms RTP frames, drop some, and let the receiver conceal them.

      "plc"   repeat the last good frame, 3 dB quieter per repeat, cross-faded at the splice. This is the idea
              behind G.711 Appendix I packet loss concealment, without its pitch analysis.
      "zero"  digital silence: a receiver with no concealment, the classic telephone dropout.

    Returns (signal, realised loss fraction) - the realised fraction is recorded per call, because a Markov
    chain over a few thousand frames does not land exactly on the requested rate.
    """
    n_frames = int(np.ceil(len(wave) / FRAME_SAMPLES))
    padded = np.pad(wave, (0, n_frames * FRAME_SAMPLES - len(wave)))
    frames = padded.reshape(n_frames, FRAME_SAMPLES).copy()
    lost = gilbert_elliott(n_frames, loss_rate, mean_burst, rng)
    run = 0
    for i in range(n_frames):
        if not lost[i]:
            run = 0
            continue
        run += 1
        if concealment == "zero" or i == 0:
            frames[i] = 0.0
        else:
            frames[i] = frames[i - 1] * (10 ** (-3.0 * run / 20))
            ramp = np.linspace(0, 1, 16, dtype=np.float32)          # de-click the splice
            frames[i][:16] = frames[i - 1][-16:] * (1 - ramp) + frames[i][:16] * ramp
    return frames.reshape(-1)[:len(wave)].astype(np.float32), float(lost.mean())


def clock_drift(wave, ppm):
    """
    The sender's 8 kHz clock and the receiver's jitter buffer never run at exactly the same rate; the buffer
    absorbs the difference by inserting or dropping samples. A few hundred ppm of resampling is the result.
    The length is restored so the file stays aligned with the agent channel.
    """
    if abs(ppm) < 1:
        return np.asarray(wave, dtype=np.float32)
    n_out = max(8, int(round(len(wave) * (1 + ppm / 1e6))))
    drifted = np.interp(np.linspace(0, len(wave) - 1, n_out), np.arange(len(wave)), wave).astype(np.float32)
    if len(drifted) >= len(wave):
        return drifted[:len(wave)]
    return np.pad(drifted, (0, len(wave) - len(drifted)))


def comfort_noise(wave, speech_mask, level_db, rng):
    """
    Voice activity detection at the far end (DTX) stops transmitting during silence, and the receiver fills the
    gap with synthetic comfort noise at a reported level. The original room tone is gone, replaced by
    band-limited noise - which is exactly the kind of cue a detector must not be relying on.
    """
    noise = band_limit(rng.standard_normal(len(wave)).astype(np.float32), 250, 3600)
    noise *= 10 ** (level_db / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    out = np.asarray(wave, dtype=np.float32).copy()
    out[~speech_mask] = noise[~speech_mask]
    return out


def line_noise(wave, snr_db, colour, hum_hz, hum_db, speech_rms, rng):
    """Additive line noise at a given SNR relative to the speech level, plus optional mains hum."""
    noise = rng.standard_normal(len(wave)).astype(np.float32)
    if colour == "pink":                                # 1/f spectrum by FFT shaping
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise), 1 / SR)
        spec[1:] /= np.sqrt(freqs[1:])
        noise = np.fft.irfft(spec, n=len(noise)).astype(np.float32)
    noise *= speech_rms / (np.sqrt(np.mean(noise ** 2)) + 1e-12) / (10 ** (snr_db / 20))
    out = np.asarray(wave, dtype=np.float32) + noise
    if hum_hz and hum_db > -90:
        t = np.arange(len(wave)) / SR
        amp = speech_rms * 10 ** (hum_db / 20)
        out = out + (amp * (np.sin(2 * np.pi * hum_hz * t) + 0.3 * np.sin(2 * np.pi * 2 * hum_hz * t))).astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ============================================================================= alignment
def estimate_delay(reference, received, max_lag_s=0.25, mask=None):
    """
    The lag (in samples) that best aligns `received` with `reference`, from the normalised cross-correlation of
    the two ENVELOPES. Envelopes, not raw samples, because a codec rewrites the waveform - AMR-NB at 4.75 kbit/s
    barely correlates sample-for-sample with its input while its envelope still lines up perfectly.

    `mask` restricts the correlation to the samples where the ORIGINAL has speech, and it matters a lot. The
    Altur caller channel is digitally silent between turns; a telephone line is not, so after the line the
    silences carry line noise that an AGC then amplifies by up to 20 dB. Correlating over the whole file
    therefore compares "speech plus zeros" against "speech plus loud noise" and reports 0.05 for a pair that
    is aligned to the sample. Masked to the speech, the same pair reports 0.41-0.99.

    The envelope is |x| smoothed over 20 ms. Without the smoothing the correlation is dominated by the fine
    structure inside each pitch period, which the codec does rewrite, and the confidence number comes out
    meaninglessly low (0.1-0.6) even for a perfectly aligned pair.

    Sign convention: a POSITIVE lag means `received` is LATE by that many samples, which is the only case a
    causal channel can produce. (The FFT correlation c[k] = sum_m a[m+k] b[m] peaks at k = -delay, so the
    index of the peak is negated before it is returned - getting this backwards misaligns every sample by
    twice the codec delay while still looking plausible.)

    The search is deliberately limited to +-250 ms. Codec algorithmic delay is at most ~30 ms and a jitter
    buffer adds tens more, so nothing beyond that is physical - and a wider window is actively harmful,
    because speech has a syllable rhythm and the correlation will happily lock onto the WRONG period a few
    hundred milliseconds away (a 1 s window mis-aligned a rhythmic test signal by 224 ms).
    Returns (lag_samples, correlation at that lag).
    """
    n = min(len(reference), len(received))
    if n < 64:
        return 0, 0.0
    win = np.ones(FRAME_SAMPLES) / FRAME_SAMPLES
    a = np.convolve(np.abs(np.asarray(reference[:n], dtype=np.float64)), win, mode="same")
    b = np.convolve(np.abs(np.asarray(received[:n], dtype=np.float64)), win, mode="same")
    if mask is not None and mask[:n].any():
        m = np.asarray(mask[:n], dtype=bool)
        a = np.where(m, a - a[m].mean(), 0.0)
        b = np.where(m, b - b[m].mean(), 0.0)
    else:
        a -= a.mean()
        b -= b.mean()
    max_lag = min(int(max_lag_s * SR), n - 1)
    fft_n = 1 << int(np.ceil(np.log2(n + max_lag)) + 1)
    corr = np.fft.irfft(np.fft.rfft(a, fft_n) * np.conj(np.fft.rfft(b, fft_n)), fft_n)
    corr = np.concatenate([corr[-max_lag:], corr[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    k = int(np.argmax(corr))
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-12
    return int(-lags[k]), float(corr[k] / denom)


def align_to(reference, received, max_lag_s=0.25, mask=None):
    """Shift `received` so it lines up with `reference`, padded/trimmed to the reference length."""
    lag, corr = estimate_delay(reference, received, max_lag_s, mask)
    if lag > 0:                                          # received is late -> drop its first `lag` samples
        shifted = received[lag:]
    elif lag < 0:
        shifted = np.pad(received, (-lag, 0))
    else:
        shifted = received
    if len(shifted) < len(reference):
        shifted = np.pad(shifted, (0, len(reference) - len(shifted)))
    return shifted[:len(reference)].astype(np.float32), lag, corr


# ============================================================================= channel sampling
SEEN_CODECS = ["g711_ulaw", "g711_alaw", "g726_32k", "g726_24k", "g726_16k"]
UNSEEN_CODECS = ["gsm_fr", "amrnb_12k2", "amrnb_7k4", "amrnb_4k75", "ilbc", "speex", "opus_nb", "g722"]
CODEC_FAMILY = {**{c: "seen" for c in SEEN_CODECS}, **{c: "unseen" for c in UNSEEN_CODECS}}
FAMILIES = {"seen": SEEN_CODECS, "unseen": UNSEEN_CODECS}


def sample_channel(rng, family="seen", severity=1.0):
    """
    Draw one telephone channel. `family` picks the codec pool ("seen" may be trained on, "unseen" may not);
    `severity` scales the impairments (1.0 = the ranges below, which bracket what carriers actually deliver).

    Everything comes from a seeded numpy Generator, so a (call id, family, variant) triple always produces the
    same channel: the dataset is reproducible and regenerating it is idempotent.
    """
    pool = FAMILIES[family]
    n_hops = 1 if rng.random() < 0.65 else 2             # a call crossing two carriers is transcoded twice
    return {
        "family": family,
        "codecs": [str(c) for c in rng.choice(pool, size=n_hops, replace=True)],
        "high_pass_hz": float(rng.uniform(180, 380)),
        "low_pass_hz": float(rng.uniform(3000, 3800)),
        "tilt_db_oct": float(rng.uniform(-2.5, 1.5) * severity),
        "agc": bool(rng.random() < 0.75),
        "agc_target_db": float(rng.uniform(-26, -16)),
        "input_gain_db": float(rng.uniform(-10, 4) * severity),
        "loss_rate": float(rng.choice([0.0, 0.0, 0.005, 0.01, 0.02, 0.04]) * severity),
        "loss_burst": float(rng.uniform(1.0, 4.0)),
        "concealment": str(rng.choice(["plc", "plc", "zero"])),
        "clock_drift_ppm": float(rng.uniform(-300, 300) * severity),
        "comfort_noise": bool(rng.random() < 0.5),
        "comfort_noise_db": float(rng.uniform(-62, -45)),
        "noise_snr_db": float(rng.uniform(18, 42)),
        "noise_colour": str(rng.choice(["white", "pink"])),
        "hum_hz": float(rng.choice([0.0, 0.0, 0.0, 50.0, 60.0])),
        "hum_db": float(rng.uniform(-50, -34)),
        "severity": float(severity),
    }


def apply_channel(wave, params, speech_mask=None, seed=0):
    """
    Push one 8 kHz float32 caller channel through the telephone line described by `params`.

    The order follows the real signal path:
        handset gain -> analogue band limit + tilt -> codec hop(s) -> RTP framing, loss, concealment
        -> jitter-buffer clock drift -> DTX comfort noise -> line noise -> AGC at the receiving gateway
    Finally the result is cross-correlated against the input and shifted back, so the transformed caller
    channel stays sample-aligned with the untouched agent channel.

    Returns (wave_out, info) where info carries the measured delay, alignment correlation, realised packet
    loss and level change - the numbers the dataset report is built from.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(wave, dtype=np.float32)
    info = {"input_rms_db": float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))}

    y = x * float(10 ** (params["input_gain_db"] / 20))
    y = band_limit(y, params["high_pass_hz"], params["low_pass_hz"])
    y = spectral_tilt(y, params["tilt_db_oct"])

    for name in params["codecs"]:
        y = codec_roundtrip(y, name)
        if len(y) != len(x):                             # codecs pad to their frame size; re-fit before the next hop
            y = y[:len(x)] if len(y) > len(x) else np.pad(y, (0, len(x) - len(y)))

    y, realised_loss = packet_loss(y, params["loss_rate"], params["loss_burst"], params["concealment"], rng)
    info["realised_loss"] = realised_loss
    y = clock_drift(y, params["clock_drift_ppm"])

    if params["comfort_noise"] and speech_mask is not None:
        y = comfort_noise(y, speech_mask, params["comfort_noise_db"], rng)

    voiced = x[speech_mask] if speech_mask is not None and speech_mask.any() else x
    speech_rms = float(np.sqrt(np.mean(voiced ** 2)) + 1e-12)
    y = line_noise(y, params["noise_snr_db"], params["noise_colour"], params["hum_hz"], params["hum_db"], speech_rms, rng)

    if params["agc"]:
        y = agc(y, params["agc_target_db"])

    y, lag, corr = align_to(x, y, mask=speech_mask)
    # Residual lag after the shift: this, not the correlation, is the number that says whether the transformed
    # caller can be dropped back into channel 0 of a stereo file next to the untouched agent. An AGC legitimately
    # flattens the envelope and drags the correlation down while leaving the timing perfect.
    residual_lag, residual_corr = estimate_delay(x, y, mask=speech_mask)
    info.update({"estimated_delay_samples": int(lag), "estimated_delay_ms": 1000.0 * lag / SR,
                 "alignment_correlation": corr, "residual_delay_samples": int(residual_lag),
                 "residual_delay_ms": 1000.0 * residual_lag / SR, "residual_correlation": residual_corr,
                 "output_rms_db": float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12))})
    info["gain_db"] = info["output_rms_db"] - info["input_rms_db"]
    return y, info


def speech_mask_from_regions(regions, n_samples):
    """(start_s, end_s) speech regions -> per-sample boolean mask, for the DTX / comfort-noise stage."""
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in regions:
        mask[max(0, int(start * SR)):min(n_samples, int(np.ceil(end * SR)))] = True
    return mask
