"""Loads .env once (no dependency) and verifies the keys the service cannot run without."""
import os, pathlib

# This module used to read only the .env beside it. It now falls back to the one at the repository
# root, so the whole project can be configured from a single file - which is where the keys live when
# this module is checked out as part of the monorepo rather than on its own. A local semantic/.env
# still wins if you want to keep this service's credentials separate; setdefault gives first-wins.
_HERE = pathlib.Path(__file__).parent
for _ENV in (_HERE / ".env", _HERE.parent / ".env"):
    for _l in (_ENV.read_text(encoding="utf-8").splitlines() if _ENV.exists() else []):
        _k, _, _v = _l.partition("=")
        if _k.strip() and not _l.lstrip().startswith("#"):
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

REQUIRED = ("ELEVENLABS_API_KEY", "GEMINI_API_KEY")


def keys_present():
    return {k: bool(os.environ.get(k)) for k in REQUIRED}


def require_keys():
    missing = [k for k, ok in keys_present().items() if not ok]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}: copy .env.example to .env and fill them in")
