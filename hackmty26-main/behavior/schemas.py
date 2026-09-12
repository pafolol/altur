"""HTTP schemas; debug features belong to the internal Python interface."""

from pydantic import BaseModel, ConfigDict, Field


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_base64: str = Field(min_length=1)


class BehaviorResponse(BaseModel):
    synthetic_probability: float = Field(ge=0, le=1)
    behavior_confidence: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    event_count: int = Field(ge=0)


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(ge=0, le=1)
