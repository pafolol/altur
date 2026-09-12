"""Local-only Altur data access. IDs are join keys, never classifier features."""

from pathlib import Path
import hashlib
import json
import re
import wave
import numpy as np
import pandas as pd
from .config import ROOT, feature_config


def read_manifest(path=ROOT / "manifest.csv"):
    df = pd.read_csv(path)
    required = {"anon_id", "label", "split", "duration_s"}
    if not required <= set(df.columns) or df[list(required)].isna().any().any():
        raise ValueError("Manifest schema is incomplete")
    if df.anon_id.duplicated().any():
        raise ValueError("Duplicate call IDs in manifest")
    if not df.anon_id.map(lambda x: bool(re.fullmatch(r"call_[0-9a-f]+", str(x)))).all():
        raise ValueError("Unsafe or unsupported call identifier")
    if not set(df.label) <= {"human", "synthetic"} or not set(df.split) <= {"train", "val"}:
        raise ValueError("Unexpected labels/splits; official split must be preserved")
    if not np.isfinite(df.duration_s).all() or (df.duration_s <= 0).any():
        raise ValueError("Invalid manifest duration")
    df["y"] = (df.label == "synthetic").astype(int)
    return df.sort_values("anon_id").reset_index(drop=True)


def locate_audio(explicit=None):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_dir():
            raise FileNotFoundError("Configured audio directory is missing")
        return path
    candidates = [p for p in (ROOT / "audio", ROOT.parent / "audio") if p.is_dir()]
    if len(candidates) != 1:
        raise ValueError("Specify --audio-dir: expected exactly one local audio directory")
    return candidates[0]


def wav_duration(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def load_feature_table(source):
    """Reject stale schema/config, edited cache, and changed official split assignments."""
    path = ROOT / "artifacts/features" / f"{source}.csv"
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if meta["feature_config"] != feature_config():
        raise ValueError("Stale feature configuration; re-extract features")
    if meta["table_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Feature table checksum mismatch; re-extract features")
    if source == "vad":
        from .vad import MODEL_SHA256
        cfg = json.loads((ROOT / "artifacts/vad/config.json").read_text())
        if meta["vad_config"] != cfg or meta["vad_model_sha256"] != MODEL_SHA256:
            raise ValueError("VAD cache configuration mismatch; re-extract turns and features")
    df = pd.read_csv(path)
    official = read_manifest().set_index("anon_id")[["split", "y"]].sort_index()
    actual = df.set_index("anon_id")[["split", "y"]].sort_index()
    if not actual.equals(official):
        raise ValueError("Feature table must preserve all official call IDs, splits and labels")
    return df
