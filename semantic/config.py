"""Loads .env once (no dependency) and verifies the keys the service cannot run without."""
import os, pathlib

_ENV = pathlib.Path(__file__).parent / ".env"
for _l in (_ENV.read_text().splitlines() if _ENV.exists() else []):
    _k, _, _v = _l.partition("=")
    if _k.strip() and not _l.lstrip().startswith("#"):
        os.environ.setdefault(_k.strip(), _v.strip())

REQUIRED = ("ELEVENLABS_API_KEY", "GEMINI_API_KEY")


def keys_present():
    return {k: bool(os.environ.get(k)) for k in REQUIRED}


def require_keys():
    missing = [k for k, ok in keys_present().items() if not ok]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}: copy .env.example to .env and fill them in")
