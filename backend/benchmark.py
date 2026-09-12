import argparse
import base64
import math
import statistics
import time
from pathlib import Path

import requests


parser = argparse.ArgumentParser(description="Benchmark del endpoint /detect")
parser.add_argument("folder", type=Path)
parser.add_argument("--url", default="http://127.0.0.1:8000/detect")
args = parser.parse_args()
latencies = []; successful = failed = 0
for path in sorted(args.folder.glob("*.wav")):
    try:
        payload = {"audio": base64.b64encode(path.read_bytes()).decode("ascii")}
        started = time.perf_counter(); response = requests.post(args.url, json=payload, timeout=120); elapsed = time.perf_counter() - started
        if response.ok: successful += 1; latencies.append(elapsed)
        else: failed += 1
    except requests.RequestException: failed += 1
if latencies:
    ordered = sorted(latencies); p95 = ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))]
    print(f"requests exitosas: {successful}\nrequests fallidas: {failed}\nmean latency: {statistics.mean(latencies):.3f} s\nmedian latency: {statistics.median(latencies):.3f} s\np95 latency: {p95:.3f} s\nmax latency: {max(latencies):.3f} s")
else:
    print(f"requests exitosas: {successful}\nrequests fallidas: {failed}\nNo hay latencias exitosas")
