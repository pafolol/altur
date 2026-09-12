"""Strict PCM WAV input. Never downmix the two participant channels."""

import io
import wave
from dataclasses import dataclass
import numpy as np


class AudioError(ValueError):
    pass


@dataclass
class Audio:
    samples: np.ndarray  # [samples, channels], float32 in [-1,1]
    sample_rate: int = 8000

    @property
    def duration(self):
        return len(self.samples) / self.sample_rate


def load_wav(wav_bytes: bytes, max_duration_s: float = 600.) -> Audio:
    """Reject wrong rate/encoding/channel count, empty and truncated inputs explicitly."""
    if not isinstance(wav_bytes, bytes):
        raise AudioError("Expected WAV bytes")
    if len(wav_bytes) > int(max_duration_s * 8000 * 4) + 65536:
        raise AudioError("WAV exceeds the configured size limit")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (2, 8000, 2, "NONE"):
                raise AudioError("Expected stereo 8000 Hz 16-bit PCM WAV; caller channel 0, agent channel 1")
            n = wav.getnframes()
            if n <= 0 or n / 8000 > max_duration_s:
                raise AudioError("Empty or over-duration WAV")
            payload = wav.readframes(n)
            if len(payload) != n * 4:
                raise AudioError("Truncated WAV sample data")
    except (wave.Error, EOFError, OverflowError, RuntimeError) as exc:
        # Python's wave._Chunk raises RuntimeError for an out-of-bounds RIFF seek.
        # This scope contains only container parsing/reading, not VAD inference.
        raise AudioError("Invalid WAV container") from exc
    return Audio(np.frombuffer(payload, dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768.)


def encode_wav(samples: np.ndarray) -> bytes:
    """Utility for tests and timing-preserving robustness experiments."""
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes((np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2").tobytes())
    return stream.getvalue()
