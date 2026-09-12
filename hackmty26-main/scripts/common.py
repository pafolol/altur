"""Small experiment bookkeeping helpers. Never print per-call records."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from behavior.config import ROOT, SEED, feature_config


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance():
    commit = None
    if (ROOT / ".git").exists():
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        commit = proc.stdout.strip() if proc.returncode == 0 else None
    source = hashlib.sha256()
    for folder in ("behavior", "scripts"):
        for file in sorted((ROOT / folder).glob("*.py")):
            source.update(file.name.encode())
            source.update(file.read_bytes())
    return {"date_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
            "python": platform.python_version(), "platform": platform.platform(),
            "git_commit": commit, "source_sha256": source.hexdigest(),
            "feature_config": feature_config(),
            "packages": {n: importlib.metadata.version(n) for n in
                         ("numpy", "scipy", "scikit-learn", "pandas", "onnxruntime", "fastapi")}}
