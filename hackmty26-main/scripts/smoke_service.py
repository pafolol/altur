"""Exercise the actual loopback HTTP service; no request leaves this computer."""

import argparse
import base64
import csv
import json
import os
import socket
import subprocess
import sys
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from behavior.config import ROOT
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Interpreter with runtime dependencies only")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/metrics/http_smoke.json")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    start = perf_counter()
    proc = subprocess.Popen([args.python, "-m", "uvicorn", "behavior.service:app", "--host", "127.0.0.1",
                             "--port", str(port), "--no-access-log"], cwd=ROOT,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            if proc.poll() is not None:
                raise RuntimeError("Local service failed to start: " + proc.stderr.read().decode())
            try:
                with urlopen(base + "/health", timeout=.2) as response:
                    health = json.load(response)
                break
            except (URLError, TimeoutError):
                sleep(.1)
        else:
            raise RuntimeError("Service readiness timed out")
        startup = perf_counter() - start
        with (ROOT / "manifest.csv").open() as f:
            row = next(csv.DictReader(f))
        folder = ROOT / "audio" if (ROOT / "audio").exists() else ROOT.parent / "audio"
        payload = (folder / (row["anon_id"] + ".wav")).read_bytes()
        body = json.dumps({"audio_base64": base64.b64encode(payload).decode()}).encode()
        durations = {}
        for endpoint, keys in (("behavior", {"synthetic_probability", "quality_score", "event_count", "behavior_confidence"}),
                               ("detect", {"is_synthetic", "confidence"})):
            start = perf_counter()
            req = Request(base + "/" + endpoint, data=body, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=30) as response:
                result = json.load(response)
            if set(result) != keys:
                raise AssertionError("HTTP contract mismatch")
            durations[endpoint] = perf_counter() - start
        from behavior.audio import encode_wav
        import numpy as np
        silent = json.dumps({"audio_base64": base64.b64encode(encode_wav(np.zeros((8000, 2)))).decode()}).encode()
        req = Request(base + "/detect", data=silent, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as response:
            assert response.headers["X-Behavior-Evidence"] == "insufficient_evidence"
            assert json.load(response) == {"is_synthetic": False, "confidence": .5}
        invalid = Request(base + "/detect", data=b'{"audio_base64":"CONFIDENTIAL_INVALID"}', headers={"Content-Type": "application/json"})
        try:
            urlopen(invalid, timeout=5)
            raise AssertionError("Invalid audio accepted")
        except HTTPError as error:
            assert error.code == 422 and b"CONFIDENTIAL_INVALID" not in error.read()
        summary = {"health": health["status"], "startup_to_ready_s": startup,
                   "real_wav_endpoints_passed": list(durations), "single_call_http_s": durations,
                   "invalid_input_rejected_without_echo": True,
                   "silence_nonflag_fallback_and_evidence_header": True,
                   "scope": "Actual HTTP on 127.0.0.1 only; not cloud or load testing"}
        path = args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(path.read_text()) if path.exists() else None
        history = previous.get("runs", [previous]) if previous else []
        summary["runs"] = history + [summary.copy()]
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
