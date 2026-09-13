from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import get_settings
from .schemas import TelemetryCreate


class DatabaseUnavailable(Exception):
    """Raised when Tiger Data cannot be reached or is not configured."""


def _connect() -> psycopg.Connection:
    url = get_settings().tiger_database_url
    if not url:
        raise DatabaseUnavailable("Tiger Data is not configured")
    try:
        return psycopg.connect(url, row_factory=dict_row)
    except Exception as exc:
        raise DatabaseUnavailable("Tiger Data is unavailable") from exc


def insert_detection(telemetry: TelemetryCreate) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    values = (timestamp, *telemetry.model_dump().values())
    query = """
        INSERT INTO detection_events (
            timestamp, call_id, is_synthetic, confidence, acoustic_score,
            behavioral_score, semantic_score, final_score, latency_ms
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING timestamp, call_id, is_synthetic, confidence, acoustic_score,
                  behavioral_score, semantic_score, final_score, latency_ms
    """
    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, values)
                row = cursor.fetchone()
            connection.commit()
            return dict(row)
    except DatabaseUnavailable:
        raise
    except Exception as exc:
        raise DatabaseUnavailable("Tiger Data is unavailable") from exc


def fetch_detections(limit: int = 50) -> list[dict[str, Any]]:
    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT timestamp, call_id, is_synthetic, confidence,
                              acoustic_score, behavioral_score, semantic_score,
                              final_score, latency_ms
                       FROM detection_events
                       ORDER BY timestamp DESC LIMIT %s""",
                    (limit,),
                )
                return [dict(row) for row in cursor.fetchall()]
    except DatabaseUnavailable:
        raise
    except Exception as exc:
        raise DatabaseUnavailable("Tiger Data is unavailable") from exc


def fetch_stats() -> dict[str, Any]:
    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT COUNT(*) AS total_detections,
                              COUNT(*) FILTER (WHERE is_synthetic) AS synthetic_detections,
                              COUNT(*) FILTER (WHERE NOT is_synthetic) AS human_detections,
                              AVG(confidence) AS average_confidence,
                              AVG(latency_ms) AS average_latency_ms
                       FROM detection_events"""
                )
                return dict(cursor.fetchone())
    except DatabaseUnavailable:
        raise
    except Exception as exc:
        raise DatabaseUnavailable("Tiger Data is unavailable") from exc
