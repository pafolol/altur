"""The stress-test perturbations must keep shape, level and finiteness; the low-pass must actually remove the band."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import audio  # noqa: E402
import stress_test  # noqa: E402
from test_pipeline import synthetic_call  # noqa: E402


def test_perturbations_keep_shape_and_level():
    stereo, sr = synthetic_call()
    caller = audio.get_channel(stereo, 0)
    for name, fn in stress_test.CONDITIONS.items():
        out = fn(caller)
        assert out.shape == caller.shape and out.dtype == np.float32, name
        assert np.isfinite(out).all(), name
        assert abs(audio.rms_db(out) - audio.rms_db(caller)) < 15, name


def test_lowpass_removes_band():
    stereo, sr = synthetic_call()
    caller = audio.get_channel(stereo, 0)
    lp = stress_test.lowpass(caller, 3400)
    spec = np.abs(np.fft.rfft(lp)) ** 2
    freqs = np.fft.rfftfreq(len(lp), 1 / sr)
    assert spec[freqs > 3800].sum() < 1e-3 * spec[freqs < 3000].sum()
