# Tiger Telemetry Service

Standalone FastAPI service for storing deepfake detection result telemetry in Tiger Data/PostgreSQL. It contains no detector, ML, Twilio, audio, transcript, or PII handling logic.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set `TIGER_DATABASE_URL`.
4. Run `schema.sql` against the Tiger Data PostgreSQL endpoint.

The application reads environment variables directly. If using a `.env` file locally, load it in your shell or use a process manager that loads it; no credentials are committed.

## Run

From the repository root, so it reads the project `.env` (`TIGER_DATABASE_URL` lives there):

```text
tiger-telemetry/.venv/bin/uvicorn --app-dir tiger-telemetry --env-file .env app.main:app --port 8200
```

The service is available at `http://127.0.0.1:8200` (8000 is the detector). API documentation is at `/docs`.

The detector (`src/server.py`) mirrors every verdict here on its own: set `TELEMETRY_URL=http://127.0.0.1:8200`
in the project `.env` and each call it scores arrives as one `POST /telemetry`, after its SQLite call log and
off the hot path. A failed mirror is one printed line on the detector, never a failed detection.

## Test

```text
pytest
```

Tests mock all database operations and never connect to a real database.

## Tiger Data setup

1. Create or open a Tiger Cloud service with PostgreSQL/TimescaleDB enabled.
2. Open the service's connection details or Connect panel.
3. Select the PostgreSQL connection string, using SSL if offered.
4. Copy the full URI, typically shaped like `postgresql://USER:PASSWORD@HOST:PORT/DATABASE?sslmode=require`.
5. Set it as `TIGER_DATABASE_URL` in the local `.env` or deployment secret. Do not paste it into source code.
6. Connect with `psql` or the Tiger SQL editor and execute `schema.sql`.
7. If `create_hypertable` is unavailable, run only the `CREATE TABLE` statement; the service works with a regular PostgreSQL table too.

Tiger Cloud labels and connection panels can vary by plan, so use the connection URI shown for the specific service rather than constructing credentials manually.
