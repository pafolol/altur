from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import database
from .config import get_settings
from .schemas import Detection, Stats, TelemetryCreate

settings = get_settings()
app = FastAPI(title="Tiger Telemetry Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _database_error() -> HTTPException:
    return HTTPException(status_code=503, detail="Tiger Data is unavailable")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/telemetry", response_model=Detection, status_code=201)
def create_telemetry(telemetry: TelemetryCreate) -> Detection:
    try:
        return database.insert_detection(telemetry)
    except database.DatabaseUnavailable as exc:
        raise _database_error() from exc


@app.get("/detections", response_model=list[Detection])
def detections() -> list[Detection]:
    try:
        return database.fetch_detections()
    except database.DatabaseUnavailable as exc:
        raise _database_error() from exc


@app.get("/stats", response_model=Stats)
def stats() -> Stats:
    try:
        return database.fetch_stats()
    except database.DatabaseUnavailable as exc:
        raise _database_error() from exc
