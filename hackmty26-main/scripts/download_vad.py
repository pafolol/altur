"""SETUP ONLY: download public MIT model weights. Never reads/uploads Altur data."""

import hashlib
import urllib.request
from behavior.vad import MODEL_PATH, MODEL_SHA256, VAD_COMMIT, VAD_RELEASE
from .common import write_json


def main():
    url = f"https://raw.githubusercontent.com/snakers4/silero-vad/{VAD_COMMIT}/src/silero_vad/data/silero_vad.onnx"
    if MODEL_PATH.exists():
        data = MODEL_PATH.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read(10_000_000)
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Downloaded model checksum mismatch")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        MODEL_PATH.write_bytes(data)
    write_json(MODEL_PATH.with_suffix(".json"), {"release": VAD_RELEASE, "commit": VAD_COMMIT,
               "url": url, "sha256": digest, "bytes": len(data), "license": "MIT"})
    print(f"Public Silero model {VAD_RELEASE}: {len(data)} bytes, sha256={digest}")


if __name__ == "__main__":
    main()
