"""Final WAV-only proof in a data-free package with macOS network/file sandboxing.

The parent supplies ONE local WAV through stdin. The child receives no labels,
manifest, organizer turns, training tables or cached features. No WAV is persisted.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from behavior.config import ROOT

TEMP_ROOT = Path("/var/folders/l6/p5dgmv7x2653mh07zb0gst3w0000gn/T/opencode")
CHILD = r'''
import sys, json, importlib.util, socket, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
started = time.perf_counter()
from behavior import BehaviorDetector
from behavior.audio import load_wav, encode_wav
from behavior.decision import challenge_response
import numpy as np

payload = sys.stdin.buffer.read()
detector = BehaviorDetector()
load_s = time.perf_counter() - started
assert all(importlib.util.find_spec(p) is None for p in ("sklearn", "pandas", "torch"))
original_root = Path(sys.argv[2])
file_access_blocked = False
try:
    (original_root / "manifest.csv").read_bytes()
except PermissionError:
    file_access_blocked = True
assert file_access_blocked
network_blocked = False
try:
    with socket.socket() as s:
        s.connect(("127.0.0.1", 9))
except PermissionError:
    network_blocked = True
assert network_blocked
audio = load_wav(payload)
start = time.perf_counter()
full = detector.predict(payload)
valid_s = time.perf_counter() - start
assert full["event_count"] >= 2 and full["quality_score"] > 0
assert 0 <= full["synthetic_probability"] <= 1
snippet = audio.samples[:min(len(audio.samples), 30 * 8000)]
caller_only, agent_only = snippet.copy(), snippet.copy()
caller_only[:, 1] = 0
agent_only[:, 0] = 0
sparse = np.zeros((3 * 8000, 2), dtype=np.float32)
sparse[8000:8800] = snippet[:800]
fixtures = {"silence": np.zeros((8000, 2), dtype=np.float32),
            "sparse_100ms_burst": sparse, "caller_only": caller_only, "agent_only": agent_only}
checks = {}
for name, waveform in fixtures.items():
    result = detector.predict(encode_wav(waveform), debug=True)
    assert result["quality_score"] == 0 and result["synthetic_probability"] == .5
    assert result["diagnostics"]["evidence_state"] == "insufficient_evidence"
    assert challenge_response(result, detector.threshold) == {"is_synthetic": False, "confidence": .5}
    checks[name] = "passed: abstention and non-flag Boolean"
after = detector.predict(payload)
assert abs(after["synthetic_probability"] - full["synthetic_probability"]) < 1e-12
print(json.dumps({"valid_wav_inference": "passed", "training_data_access_denied": file_access_blocked,
                  "network_egress_denied": network_blocked, "native_process_sandbox": True,
                  "no_sklearn_pandas_torch": True, "insufficient_evidence_fixtures": checks,
                  "state_isolation_after_different_calls": "passed",
                  "import_and_model_load_s": load_s, "single_valid_wav_inference_s": valid_s,
                  "python": sys.version.split()[0]}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports/metrics/delta_review/isolated_runtime.json"))
    args = parser.parse_args()
    runtime = ROOT / ".venv-runtime/bin/python"
    sandbox = shutil.which("sandbox-exec")
    if not runtime.is_file() or sandbox is None:
        raise SystemExit("Preconfigured runtime/macOS sandbox unavailable; do not claim isolated verification")
    audio_dir = ROOT / "audio" if (ROOT / "audio").is_dir() else ROOT.parent / "audio"
    payload = next(iter(sorted(audio_dir.glob("*.wav")))).read_bytes()
    artifact_hash = hashlib.sha256((ROOT / "artifacts/models/behavior.json").read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="behavior-delta-runtime-", dir=TEMP_ROOT) as temp:
        package = Path(temp)
        (package / "behavior").mkdir()
        for path in (ROOT / "behavior").glob("*.py"):
            shutil.copy2(path, package / "behavior" / path.name)
        for relative in ("artifacts/models/behavior.json", "artifacts/vad/silero_vad.onnx"):
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        # Deny original project/data. Allow only its existing interpreter + site-packages.
        # Native ONNX operations receive the same OS-enforced network restriction.
        def quote(path):
            return json.dumps(str(path.resolve()))
        profile = (f'(version 1) (allow default) (deny network*) '
                   f'(deny file-read* (subpath {quote(ROOT.parent)})) '
                   f'(allow file-read* (subpath {quote(ROOT / ".venv-runtime")}) '
                   f'(subpath {quote(ROOT / ".python")}))')
        environment = {k: v for k, v in os.environ.items() if not k.startswith(("BEHAVIOR_", "PYTHONPATH"))}
        environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run([sandbox, "-p", profile, str(runtime), "-I", "-B", "-c", CHILD,
                                    str(package), str(ROOT)], cwd=package, env=environment, input=payload,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        if completed.returncode:
            raise RuntimeError("Isolated runtime check failed: " + completed.stderr.decode())
        result = json.loads(completed.stdout)
        result["artifact_sha256"] = artifact_hash
        result["package_contents"] = "runtime .py files + Behaviour JSON + Silero ONNX only; audio bytes supplied on stdin"
        result["docker_status"] = "NOT VERIFIED LOCALLY"
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
