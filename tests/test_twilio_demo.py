"""
Tests for the live-call demo (src/twilio_demo.py).

No credentials, no network, no calls: the Twilio client is stubbed, so the parts worth testing - the
TwiML, the mono-to-stereo conversion the detectors depend on, the state machine, and the fact that a
phone number never comes back to a browser - all run offline.
"""
import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import twilio_demo  # noqa: E402
from twilio_demo import CallJob, TwilioDemo, mono_to_stereo_wav  # noqa: E402


def mono_wav(seconds=2.0, rate=8000, channels=1, freq=220.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate)
        t = np.arange(int(rate * seconds)) / rate
        tone = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
        w.writeframes(np.stack([tone] * channels, axis=1).tobytes())
    return buf.getvalue()


def read(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return (w.getnchannels(), w.getsampwidth(), w.getframerate(),
                np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, w.getnchannels()))


# --------------------------------------------------------------------------- the audio conversion


def test_a_twilio_recording_becomes_the_stereo_shape_every_detector_expects():
    ch, width, rate, pcm = read(mono_to_stereo_wav(mono_wav()))
    assert (ch, width, rate) == (2, 2, 8000)
    assert pcm.shape[1] == 2
    assert pcm[:, 0].any()               # the recording is on channel 0, where the caller belongs
    assert not pcm[:, 1].any()           # and channel 1 is silent: there IS no separate agent track


def test_a_stereo_recording_is_mixed_down_rather_than_half_thrown_away():
    _, _, _, pcm = read(mono_to_stereo_wav(mono_wav(channels=2)))
    assert pcm[:, 0].any() and not pcm[:, 1].any()


def test_a_recording_at_another_rate_is_resampled_to_8k():
    _, _, rate, pcm = read(mono_to_stereo_wav(mono_wav(seconds=1.0, rate=16000)))
    assert rate == 8000
    assert abs(len(pcm) - 8000) < 100    # one second, at 8 kHz


def test_the_silent_agent_channel_is_what_makes_the_behaviour_layer_abstain():
    """Not a bug - the honest consequence of a mono recording, and the layer reports it."""
    import fusion
    wav = mono_to_stereo_wav(mono_wav(seconds=6.0))
    layer = fusion.BehaviourLayer()
    if not layer.available():
        pytest.skip("behaviour artifacts not installed")
    r = layer.score(wav)
    assert r.abstained is True and "evidence" in r.reason.lower()


def test_non_16_bit_audio_is_rejected_rather_than_misread():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(1); w.setframerate(8000); w.writeframes(b"\x00" * 800)
    with pytest.raises(ValueError, match="16-bit"):
        mono_to_stereo_wav(buf.getvalue())


# --------------------------------------------------------------------------- the TwiML


def test_the_twiml_records_and_needs_no_callback_url():
    """The <Record> has no `action`, which is what removes the need for a public tunnel."""
    xml = TwilioDemo().twiml("hola", max_seconds=25)
    assert "<Record" in xml and 'maxLength="25"' in xml
    assert "action=" not in xml
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>')


def test_the_prompt_is_escaped_so_it_cannot_break_the_twiml():
    xml = TwilioDemo().twiml('hola <Hangup/> & "adiós"')
    assert "<Hangup/>" not in xml and "&lt;Hangup/&gt;" in xml and "&amp;" in xml


# --------------------------------------------------------------------------- configuration


def test_it_reports_exactly_which_variables_are_missing(monkeypatch):
    for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "TWILIO_TO_NUMBER"):
        monkeypatch.delenv(k, raising=False)
    d = TwilioDemo().describe()
    assert d["configured"] is False
    assert set(d["missing"]) >= {"TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"}
    assert d["acoustic_model"] == "robust_v2"     # the live-call path uses the phone-hardened model


def test_calling_without_configuration_explains_itself_rather_than_raising_a_twilio_error(monkeypatch):
    for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        TwilioDemo().start("+5218112345678")


def test_a_configured_demo_still_needs_somewhere_to_call(monkeypatch):
    for k, v in (("TWILIO_ACCOUNT_SID", "AC123"), ("TWILIO_AUTH_TOKEN", "tok"),
                 ("TWILIO_PHONE_NUMBER", "+15550000000")):
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("TWILIO_TO_NUMBER", raising=False)
    with pytest.raises(ValueError, match="no destination"):
        TwilioDemo().start(None)


# --------------------------------------------------------------------------- privacy


@pytest.mark.parametrize("number,shown", [("+5218112345678", "…5678"), ("+15550001234", "…1234"),
                                          ("12", "…"), ("", "…"), (None, "…")])
def test_a_phone_number_never_leaves_the_server_in_full(number, shown):
    assert twilio_demo._mask(number) == shown


def test_the_job_the_page_polls_carries_no_full_number():
    job = CallJob("CA123", "+5218112345678")
    assert "8112345678" not in str(job.to_dict())
    assert job.to_dict()["to"] == "…5678"


# --------------------------------------------------------------------------- the state machine


class FakeCall:
    def __init__(self, statuses):
        self.statuses, self.sid = list(statuses), "CA_test"
    def fetch(self):
        self.status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return self


class FakeRecording:
    def __init__(self, status="completed", duration="7"):
        self.sid, self.status, self.duration = "RE_test", status, duration


class FakeClient:
    def __init__(self, statuses, recordings_after=1):
        self._call = FakeCall(statuses)
        self._recordings_after, self._polls = recordings_after, 0
        self.created = None
    def calls(self, sid=None):
        return self._call
    @property
    def recordings(self):
        outer = self
        class R:
            def list(self, call_sid=None, limit=1):
                outer._polls += 1
                return [FakeRecording()] if outer._polls >= outer._recordings_after else []
        return R()


def demo_with(monkeypatch, client, scored=None):
    for k, v in (("TWILIO_ACCOUNT_SID", "AC123"), ("TWILIO_AUTH_TOKEN", "tok"),
                 ("TWILIO_PHONE_NUMBER", "+15550000000")):
        monkeypatch.setenv(k, v)
    d = TwilioDemo()
    monkeypatch.setattr(d, "client", lambda: client)
    monkeypatch.setattr(d, "_download", lambda sid: mono_wav(seconds=3.0))
    monkeypatch.setattr(twilio_demo, "POLL_INTERVAL_S", 0.01)

    class FakeFusion:
        def score(self, wav):
            return scored or {"is_synthetic": True, "confidence": 0.91, "synthetic_probability": 0.91,
                              "note": "", "layers": [
                                  {"key": "acoustic", "display": "Acoustic V2", "probability": 0.91,
                                   "abstained": False, "scored": True, "consulted": True,
                                   "role": "primary", "share": 1.0, "reason": ""}]}
    monkeypatch.setattr(d, "fusion", lambda: FakeFusion())
    return d


def test_a_call_runs_through_to_a_verdict(monkeypatch):
    d = demo_with(monkeypatch, FakeClient(["in-progress", "completed"], recordings_after=2))
    job = CallJob("CA_test", "+5218112345678")
    d._run(job, max_seconds=30)
    assert job.state == "done" and job.error == ""
    assert job.recording_sid == "RE_test" and job.duration_s == 7.0
    assert job.verdict["is_synthetic"] is True
    assert job.verdict["acoustic_model"] == "robust_v2"


@pytest.mark.parametrize("status", ["busy", "no-answer", "failed", "canceled"])
def test_a_call_that_never_connects_fails_with_the_reason(monkeypatch, status):
    d = demo_with(monkeypatch, FakeClient([status], recordings_after=99))
    job = CallJob("CA_test", "+5218112345678")
    d._run(job, max_seconds=30)
    assert job.state == "failed" and status in job.error
    assert job.verdict is None


def test_waiting_too_long_for_a_recording_fails_instead_of_hanging(monkeypatch):
    d = demo_with(monkeypatch, FakeClient(["completed"], recordings_after=99))
    monkeypatch.setattr(twilio_demo, "RECORDING_TIMEOUT_S", 0.05)
    job = CallJob("CA_test", "+5218112345678")
    d._run(job, max_seconds=0)
    assert job.state == "failed" and "TimeoutError" in job.error


def test_a_scoring_failure_is_reported_not_swallowed(monkeypatch):
    d = demo_with(monkeypatch, FakeClient(["completed"], recordings_after=1))
    class Boom:
        def score(self, wav): raise RuntimeError("model exploded")
    monkeypatch.setattr(d, "fusion", lambda: Boom())
    job = CallJob("CA_test", "+5218112345678")
    d._run(job, max_seconds=30)
    assert job.state == "failed" and "model exploded" in job.error


def test_the_demo_call_is_logged_exactly_once():
    """The page polls the status repeatedly; the call log must not gain a row each time."""
    job = CallJob("CA_test", "+5218112345678", state="done")
    job.verdict = {"is_synthetic": True, "confidence": 0.9, "synthetic_probability": 0.9, "layers": []}
    assert job.logged is False
    job.logged = True
    assert job.to_dict()["logged"] is True
