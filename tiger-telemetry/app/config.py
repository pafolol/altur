import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    tiger_database_url: str | None
    allowed_origins: list[str]


def get_settings() -> Settings:
    raw_origins = os.getenv("TELEMETRY_ALLOWED_ORIGINS", "")
    allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return Settings(
        tiger_database_url=os.getenv("TIGER_DATABASE_URL") or None,
        allowed_origins=allowed_origins,
    )
