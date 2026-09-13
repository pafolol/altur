CREATE TABLE IF NOT EXISTS detection_events (
    timestamp TIMESTAMPTZ NOT NULL,
    call_id TEXT NOT NULL,
    is_synthetic BOOLEAN NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    acoustic_score DOUBLE PRECISION,
    behavioral_score DOUBLE PRECISION,
    semantic_score DOUBLE PRECISION,
    final_score DOUBLE PRECISION,
    latency_ms DOUBLE PRECISION
);

-- Run this only when the Tiger Data service has TimescaleDB enabled.
SELECT create_hypertable('detection_events', 'timestamp', if_not_exists => TRUE);
