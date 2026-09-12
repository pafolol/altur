"""Fresh-process import/load and CPU WAV inference. No server/cloud benchmarks implied."""

import argparse
import csv
import json
import platform
import resource
import subprocess
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-child", action="store_true")
    parser.add_argument("--calls", type=int, default=20)
    args = parser.parse_args()
    start = perf_counter()
    from behavior.inference import BehaviorDetector
    imported = perf_counter()
    detector = BehaviorDetector()
    loaded = perf_counter()
    if args.cold_child:
        print(json.dumps({"import_s": imported - start, "model_load_s": loaded - imported,
                          "total_startup_s": loaded - start}))
        return
    import numpy as np
    from behavior.config import ROOT
    # Do not import pandas/sklearn into the inference memory benchmark.
    with (ROOT / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    indices = np.random.default_rng(2026).choice(len(rows), min(args.calls, len(rows)), replace=False)
    folder = ROOT / "audio" if (ROOT / "audio").exists() else ROOT.parent / "audio"
    times = []
    for i in indices:
        r = detector.predict((folder / (rows[i]["anon_id"] + ".wav")).read_bytes(), debug=True)
        times.append(r["diagnostics"]["timing_ms"])
    startup = []
    for _ in range(3):
        start = perf_counter()
        child = subprocess.run([sys.executable, "-m", "scripts.benchmark_latency", "--cold-child"],
                               cwd=ROOT, capture_output=True, text=True, check=True)
        result = json.loads(child.stdout)
        result["process_wall_s"] = perf_counter() - start
        startup.append(result)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 ** 2) if platform.system() == "Darwin" else peak / 1024
    summary = {"calls": len(times), "cpu": platform.processor() or platform.machine(),
               "platform": platform.platform(), "python": platform.python_version(), "ort_threads": 1,
               "startup_fresh_processes": startup, "peak_process_rss_mib": peak_mb,
               "components_ms": {key: {"mean": float(np.mean([r[key] for r in times])),
                                        "p50": float(np.median([r[key] for r in times])),
                                        "p95": float(np.quantile([r[key] for r in times], .95))} for key in times[0]},
               "scope": "Local offline warmed filesystem; model-only JSON plus CPU ONNX; excludes HTTP/network transfer and base64 parsing",
               "docker_tested": False, "docker_reason": "Docker executable/runtime not installed on local machine"}
    path = ROOT / "reports/metrics/latency.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
