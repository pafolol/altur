from typing import Optional

from pydantic import BaseModel, Field, model_validator


class DetectRequest(BaseModel):
    audio: Optional[str] = None
    audio_base64: Optional[str] = None

    @model_validator(mode="after")
    def has_audio(self):
        if not (self.audio or self.audio_base64):
            raise ValueError("Se requiere el campo 'audio' o 'audio_base64'")
        return self


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(ge=0.0, le=1.0)
