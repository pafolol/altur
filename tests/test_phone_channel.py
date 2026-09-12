"""
Unit tests for the telephone channel.

The expensive mistakes this file exists to catch:
  * a G.711 table that is subtly wrong (silently degrading every sample in the dataset)
  * mu-law bytes read as if they were PCM16 - the classic telephony bug, which would invalidate the entire
    dataset while still producing a file that plays
  * a codec that is secretly a no-op, so the "channel" changes nothing
  * an alignment estimator that cannot find a delay it was handed on a plate
  * a channel draw that is not reproducible from its seed
"""
import audioop
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import phone_channel as pc  # noqa: E402

SR = pc.SR


def speech_like(seconds=2.0, seed=0):
    """
    A voiced test signal: a harmonic stack with a wandering pitch, shaped by three formant resonances and a
    syllable-rate envelope, with a pause in it.

    The formants matter. GSM, AMR and iLBC are CELP codecs - they fit an all-pole vocal-tract model to the
    input, so a bare harmonic buzz falls outside what they are built to code and they mangle it far worse
    than they mangle speech. Without the resonances, iLBC scores 0.79 envelope correlation here while
    managing 0.93 on a real call from the dataset, and the test would be measuring the fixture, not the codec.
    """
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    w = sum(np.sin(k * phase) / k for k in range(1, 12))
    for centre, bandwidth in ((700, 90), (1220, 110), (2600, 160)):   # a neutral vowel
        r = np.exp(-np.pi * bandwidth / SR)
        w = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * centre / SR), r ** 2], w)
    # An irregular syllable envelope, not a sine: a strictly periodic one makes the envelope correlation
    # ambiguous and any aligner can lock onto the wrong period.
    knots = np.arange(0, seconds + 0.36, 0.18)
    w *= np.interp(t, knots, rng.uniform(0.15, 1.0, len(knots))) ** 2
    w += 0.01 * rng.standard_normal(len(t))
    w[int(0.4 * SR):int(0.7 * SR)] = 0.0                              # a pause, like a real caller
    return (0.4 * w / (np.abs(w).max() + 1e-9)).astype(np.float32)


# ----------------------------------------------------------------------------- G.711
def test_ulaw_encode_matches_reference_over_every_int16():
    """audioop is the reference C implementation of G.711. Anything less than exact is a bug."""
    vals = np.arange(-32768, 32768, dtype=np.int16)
    reference = np.frombuffer(audioop.lin2ulaw(vals.tobytes(), 2), dtype=np.uint8)
    assert np.array_equal(pc.pcm16_to_ulaw(vals), reference)


def test_ulaw_decode_matches_reference_over_every_code():
    codes = np.arange(256, dtype=np.uint8)
    reference = np.frombuffer(audioop.ulaw2lin(codes.tobytes(), 2), dtype="<i2")
    assert np.array_equal(pc.ulaw_to_pcm16(codes), reference)


def test_alaw_encode_matches_reference_over_every_int16():
    vals = np.arange(-32768, 32768, dtype=np.int16)
    reference = np.frombuffer(audioop.lin2alaw(vals.tobytes(), 2), dtype=np.uint8)
    assert np.array_equal(pc.pcm16_to_alaw(vals), reference)


def test_alaw_decode_matches_reference_over_every_code():
    codes = np.arange(256, dtype=np.uint8)
    reference = np.frombuffer(audioop.alaw2lin(codes.tobytes(), 2), dtype="<i2")
    assert np.array_equal(pc.alaw_to_pcm16(codes), reference)


def test_ulaw_roundtrip_is_close_to_the_original():
    """G.711 is lossy but only just: a correct round trip stays above ~30 dB SNR on speech."""
    x = speech_like()
    y = pc.pcm16_to_float(pc.ulaw_to_pcm16(pc.pcm16_to_ulaw(pc.float_to_pcm16(x))))
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean((x - y) ** 2))
    assert snr > 30, f"mu-law round trip only reached {snr:.1f} dB SNR"


def test_mulaw_bytes_read_as_pcm16_is_obviously_broken():
    """
    THE TRAP. mu-law is one byte per sample; PCM16 is two. Reading a mu-law stream as PCM16 produces a file
    that plays, has the right duration/2, and is complete garbage. This test pins down that the mistake is
    detectable, so a future change that makes it silently cannot pass.
    """
    x = speech_like()
    codes = pc.pcm16_to_ulaw(pc.float_to_pcm16(x))
    correct = pc.pcm16_to_float(pc.ulaw_to_pcm16(codes))
    misread = pc.pcm16_to_float(np.frombuffer(codes.tobytes(), dtype="<i2"))

    assert len(misread) == len(correct) // 2, "the misread signal should be half as long - that alone is a clue"
    n = len(misread)
    corr = float(np.corrcoef(correct[:n], misread)[0, 1])
    assert abs(corr) < 0.1, f"misread mu-law correlates {corr:.3f} with the truth; the trap is not detectable"

    # Two signatures that give it away on any recording, WITHOUT needing the original to compare against -
    # the situation you are actually in when a dataset arrives and you suspect it was decoded wrongly.
    rms = lambda w: float(np.sqrt(np.mean(w ** 2)))
    kurtosis = lambda w: float(np.mean((w - w.mean()) ** 4) / (np.var(w) ** 2 + 1e-12))
    # mu-law codes spread over the whole byte range, so read as PCM16 they sit near full scale ...
    assert rms(misread) > 3 * rms(correct), \
        f"misread mu-law should be near full scale ({rms(misread):.3f} vs {rms(correct):.3f})"
    # ... and their distribution is almost uniform (kurtosis of a uniform variable is 1.8), while speech is
    # heavy-tailed: mostly near silence with occasional loud excursions.
    assert kurtosis(correct) > 3.0, f"sanity: speech should be heavy-tailed, got kurtosis {kurtosis(correct):.2f}"
    assert kurtosis(misread) < 2.5, f"misread mu-law should be near-uniform, got kurtosis {kurtosis(misread):.2f}"


# ----------------------------------------------------------------------------- codecs
@pytest.mark.parametrize("name", ["g711_ulaw", "g711_alaw", *pc.FFMPEG_CODECS])
def test_codec_roundtrip_keeps_rate_and_roughly_the_length(name):
    x = speech_like(2.0)
    y = pc.codec_roundtrip(x, name)
    assert y.dtype == np.float32
    assert abs(len(y) - len(x)) < 0.25 * SR, f"{name} changed the length by {(len(y) - len(x)) / SR:.2f} s"
    assert np.isfinite(y).all()


@pytest.mark.parametrize("name", ["g711_ulaw", "g711_alaw", *pc.FFMPEG_CODECS])
def test_codec_actually_changes_the_signal(name):
    """A 'codec' that returns its input would make the whole experiment meaningless."""
    x = speech_like(2.0)
    y = pc.codec_roundtrip(x, name)
    n = min(len(x), len(y))
    assert not np.allclose(x[:n], y[:n], atol=1e-6), f"{name} is a no-op"


@pytest.fixture(scope="module")
def real_caller_speech():
    """
    Six seconds of real caller speech from the dataset.

    The CELP codecs (GSM, AMR, iLBC) have to be tested on this and not on speech_like(): they fit an all-pole
    vocal-tract model to their input, so a synthesised buzz - however carefully shaped - falls outside what
    they are built for and they mangle its envelope (iLBC scores 0.44 on the fixture and 0.93 on a real call).
    Testing them on the fixture would measure the fixture.
    """
    import audio
    import dataset as ds
    df = ds.load_split("val")
    if df.empty or not Path(df.iloc[0]["path"]).exists():
        pytest.skip("challenge audio not available")
    stereo, sr = audio.read_wav(df.iloc[0]["path"])
    caller = audio.get_channel(stereo, 0)
    regions = audio.detect_speech_energy(caller, sr)
    speech = np.concatenate([audio.slice_chunk(caller, sr, s, e) for s, e in regions])
    return speech[:6 * SR]


@pytest.mark.parametrize("name", ["g711_ulaw", "g711_alaw", *pc.FFMPEG_CODECS])
def test_codec_still_carries_real_speech(name, real_caller_speech):
    """Lossy, but not destructive: the envelope must survive, or we are testing noise, not a phone line."""
    y, lag, corr = pc.align_to(real_caller_speech, pc.codec_roundtrip(real_caller_speech, name))
    assert corr > 0.85, f"{name} envelope correlation {corr:.3f} is too low to be a working codec"
    assert -8 <= lag <= 0.12 * SR, f"{name} delay of {1000 * lag / SR:.0f} ms is not physical for a codec"


def test_seen_and_unseen_families_are_disjoint():
    """The generalisation claim rests on this one line being true."""
    assert not set(pc.SEEN_CODECS) & set(pc.UNSEEN_CODECS)
    assert set(pc.CODEC_FAMILY) == set(pc.SEEN_CODECS) | set(pc.UNSEEN_CODECS)


# ----------------------------------------------------------------------------- impairments
def test_alignment_recovers_a_known_delay():
    x = speech_like(3.0)
    for true_lag in (0, 80, 240, -160):
        shifted = np.pad(x, (true_lag, 0))[:len(x)] if true_lag > 0 else np.pad(x, (0, -true_lag))[-true_lag:]
        lag, corr = pc.estimate_delay(x, shifted)
        assert abs(lag - true_lag) <= 8, f"expected lag {true_lag}, got {lag}"
        assert corr > 0.9


def test_packet_loss_realises_roughly_the_requested_rate():
    x = speech_like(30.0)
    rng = np.random.default_rng(0)
    _, realised = pc.packet_loss(x, 0.05, 2.0, "plc", rng)
    assert 0.02 < realised < 0.09, f"asked for 5 % loss, realised {realised:.1%}"


def test_packet_loss_zero_rate_is_a_no_op():
    x = speech_like(2.0)
    y, realised = pc.packet_loss(x, 0.0, 2.0, "plc", np.random.default_rng(0))
    assert realised == 0.0
    assert np.allclose(x, y)


def test_agc_pulls_a_quiet_and_a_loud_signal_to_the_same_place():
    """This is the mechanism that erases the 6 dB loudness gap between the dataset's two classes."""
    x = speech_like(4.0)
    quiet = pc.agc(x * 0.05, target_db=-20.0)
    loud = pc.agc(x * 0.9, target_db=-20.0)
    spread_before = abs(20 * np.log10(np.sqrt(np.mean((x * 0.05) ** 2))) - 20 * np.log10(np.sqrt(np.mean((x * 0.9) ** 2))))
    spread_after = abs(20 * np.log10(np.sqrt(np.mean(quiet ** 2))) - 20 * np.log10(np.sqrt(np.mean(loud ** 2))))
    assert spread_after < spread_before / 2, f"AGC left {spread_after:.1f} dB of a {spread_before:.1f} dB gap"


def test_comfort_noise_only_touches_the_silence():
    x = speech_like(2.0)
    mask = np.abs(x) > 1e-6
    y = pc.comfort_noise(x, mask, -55.0, np.random.default_rng(0))
    assert np.allclose(x[mask], y[mask]), "comfort noise must not touch the speech"
    assert not np.allclose(x[~mask], y[~mask]), "comfort noise must replace the silence"


def test_speech_mask_from_regions():
    mask = pc.speech_mask_from_regions([(0.0, 0.5), (1.0, 1.5)], 2 * SR)
    assert mask[:int(0.5 * SR)].all() and not mask[int(0.6 * SR):int(0.9 * SR)].any()
    assert mask[int(1.1 * SR):int(1.4 * SR)].all() and not mask[int(1.6 * SR):].any()


# ----------------------------------------------------------------------------- the whole channel
def test_channel_preserves_length_and_sample_rate():
    x = speech_like(5.0)
    params = pc.sample_channel(np.random.default_rng(1), "seen")
    y, info = pc.apply_channel(x, params, seed=1)
    assert len(y) == len(x), "the transformed caller must stay aligned with the untouched agent channel"
    assert y.dtype == np.float32 and np.isfinite(y).all()
    assert np.abs(y).max() <= 1.0
    assert "estimated_delay_ms" in info and "alignment_correlation" in info


def test_channel_is_reproducible_from_its_seed():
    x = speech_like(3.0)
    p1 = pc.sample_channel(np.random.default_rng(7), "seen")
    p2 = pc.sample_channel(np.random.default_rng(7), "seen")
    assert p1 == p2, "the same seed must draw the same channel, or the dataset is not reproducible"
    y1, _ = pc.apply_channel(x, p1, seed=3)
    y2, _ = pc.apply_channel(x, p2, seed=3)
    assert np.array_equal(y1, y2)


def test_different_seeds_draw_different_channels():
    draws = [pc.sample_channel(np.random.default_rng(s), "seen") for s in range(12)]
    assert len({tuple(d["codecs"]) + (round(d["high_pass_hz"]),) for d in draws}) > 4


@pytest.mark.parametrize("family", ["seen", "unseen"])
def test_sampled_channel_only_uses_its_own_family(family):
    for seed in range(40):
        params = pc.sample_channel(np.random.default_rng(seed), family)
        assert params["family"] == family
        for codec in params["codecs"]:
            assert pc.CODEC_FAMILY[codec] == family, f"{codec} leaked into the {family} family"


def test_channel_degrades_but_does_not_destroy():
    """Every drawn channel must still carry recognisable speech; a channel that outputs noise is a bug."""
    x = speech_like(4.0)
    for seed in range(8):
        params = pc.sample_channel(np.random.default_rng(seed), "seen")
        y, info = pc.apply_channel(x, params, seed=seed)
        assert info["alignment_correlation"] > 0.5, f"seed {seed}: correlation {info['alignment_correlation']:.2f}"
        assert not np.allclose(x, y, atol=1e-4), f"seed {seed}: the channel did nothing"
