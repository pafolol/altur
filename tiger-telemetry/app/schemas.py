from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TelemetryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1)
    is_synthetic: bool
    confidence: float = Field(ge=0, le=1)
    acoustic_score: float | None = Field(default=None, ge=0, le=1)
    behavioral_score: float | None = Field(default=None, ge=0, le=1)
    semantic_score: float | None = Field(default=None, ge=0, le=1)
    final_score: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float | None = Field(default=None, ge=0)


class Detection(TelemetryCreate):
    timestamp: datetime


class Stats(BaseModel):
    total_detections: int
    synthetic_detections: int
    human_detections: int
    average_confidence: float | None
    average_latency_ms: float | None
