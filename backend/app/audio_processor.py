import base64
import binascii
import io
import logging
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)
TARGET_SAMPLE_RATE = 8000


class AudioValidationError(ValueError):
    pass


def decode_base64_audio(value: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise AudioValidationError("El audio Base64 está vacío")
    encoded = value.strip()
    if encoded.lower().startswith("data:"):
        try:
            header, encoded = encoded.split(",", 1)
        except IndexError as exc:
            raise AudioValidationError("Data URI Base64 inválido") from exc
        if ";base64" not in header.lower():
            raise AudioValidationError("Data URI Base64 inválido")
    encoded = "".join(encoded.split())
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AudioValidationError("Base64 inválido") from exc


def load_wav_from_memory(data: bytes) -> Tuple[np.ndarray, int]:
    if not data:
        raise AudioValidationError("El audio está vacío")
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioValidationError("El WAV es inválido o no soportado") from exc
    return validate_audio(audio, sample_rate)


def validate_audio(audio: np.ndarray, sample_rate: int, max_duration: float | None = None):
    if sample_rate <= 0 or not np.isfinite(sample_rate):
        raise AudioValidationError("Sample rate inválido")
    if audio.ndim != 2 or audio.shape[1] < 2:
        raise AudioValidationError("Se requieren al menos dos canales")
    if audio.shape[0] == 0:
        raise AudioValidationError("El audio está vacío")
    duration = audio.shape[0] / float(sample_rate)
    if not np.isfinite(duration) or duration <= 0:
        raise AudioValidationError("Duración de audio inválida")
    if max_duration is not None and duration > max_duration:
        raise AudioValidationError(f"El audio supera {max_duration:g} segundos")
    if audio.shape[0] > 60_000_000:
        raise AudioValidationError("El número de muestras es demasiado grande")
    if not np.isfinite(audio).all():
        raise AudioValidationError("El audio contiene valores inválidos")
    return audio, int(sample_rate)


def resample_if_needed(audio: np.ndarray, sample_rate: int, target: int = TARGET_SAMPLE_RATE):
    if sample_rate == target:
        return audio, sample_rate
    logger.info("Resampling audio from %s Hz to %s Hz", sample_rate, target)
    channels = [librosa.resample(audio[:, i], orig_sr=sample_rate, target_sr=target) for i in range(audio.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32), target


def split_channels(audio: np.ndarray):
    if audio.ndim != 2 or audio.shape[1] < 2:
        raise AudioValidationError("Se requieren al menos dos canales")
    return audio[:, 0], audio[:, 1]
