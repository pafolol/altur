import os
from dataclasses import dataclass


def _boolean(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _weights(value: str) -> dict:
    result = {}
    for item in value.split(","):
        if item.strip():
            name, weight = item.split("=", 1)
            result[name.strip()] = float(weight)
    return result


@dataclass(frozen=True)
class Settings:
    detector_mode: str = os.getenv("DETECTOR_MODE", "mock").lower()
    model_path: str = os.getenv("MODEL_PATH", "models/detector.joblib")
    model_threshold: float = float(os.getenv("MODEL_THRESHOLD", "0.5"))
    enable_debug_endpoint: bool = _boolean(os.getenv("ENABLE_DEBUG_ENDPOINT", "false"))
    max_audio_duration_seconds: float = float(os.getenv("MAX_AUDIO_DURATION_SECONDS", "300"))
    max_request_size_mb: float = float(os.getenv("MAX_REQUEST_SIZE_MB", "20"))
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    fusion_mode: str = os.getenv("FUSION_MODE", "single_model").lower()
    fusion_weights: dict = None
    fusion_model_path: str = os.getenv("FUSION_MODEL_PATH", "models/fusion_model.joblib")
    min_audio_duration_seconds: float = float(os.getenv("MIN_AUDIO_DURATION_SECONDS", "0.25"))
    min_caller_speech_seconds: float = float(os.getenv("MIN_CALLER_SPEECH_SECONDS", "0.10"))
    enable_pitch_features: bool = _boolean(os.getenv("ENABLE_PITCH_FEATURES", "true"))
    enable_feature_warmup: bool = _boolean(os.getenv("ENABLE_FEATURE_WARMUP", "true"))

    def __post_init__(self):
        if self.fusion_weights is None:
            object.__setattr__(self, "fusion_weights", _weights(os.getenv("FUSION_WEIGHTS", "behavior=1,acoustic=1,semantic=1")))

    @property
    def max_request_bytes(self) -> int:
        return int(self.max_request_size_mb * 1024 * 1024)


settings = Settings()
