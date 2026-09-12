"""Fase 7 acceptance over real HTTP: the longest call (~11.7 MB base64) is accepted; the 71 val calls are
answered within budget; p50/p95 latency, which path was used, and the AUC of the returned scores.
  python check_server.py [--parallel N]   N concurrent requests (default 1, serial)"""
import base64, csv, subprocess, sys, time, pathlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import httpx, numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data" / "hackmty26"
rows = list(csv.DictReader(open(DATA / "manifest.csv")))
PAR = int(sys.argv[sys.argv.index("--parallel") + 1]) if "--parallel" in sys.argv else 1
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--port", "8765", "--workers", "2", "--log-level", "warning"], cwd=ROOT)
try:
    c = httpx.Client(base_url="http://127.0.0.1:8765", timeout=30)
    for _ in range(100):
        try: c.get("/docs"); break
        except httpx.ConnectError: time.sleep(0.2)
    longest = max(rows, key=lambda r: int(r["duration_s"]))
    b64 = base64.b64encode((DATA / "audio" / f"{longest['anon_id']}.wav").read_bytes()).decode()
    r = c.post("/detect", json={"audio": b64})
    print(f"longest call {longest['anon_id']} {longest['duration_s']}s payload={len(b64)/1e6:.1f} MB -> {r.status_code} {r.json()}")
    assert r.status_code == 200
    assert c.get("/health").json()["ok"]
    def one(row):
        b64 = base64.b64encode((DATA / "audio" / f"{row['anon_id']}.wav").read_bytes()).decode()
        t0 = time.perf_counter(); r = c.post("/detect", json={"audio": b64}); dt = time.perf_counter() - t0
        assert r.status_code == 200, r.text
        return dt * 1000, r.json(), row["label"] == "synthetic"
    with ThreadPoolExecutor(PAR) as ex:
        res = list(ex.map(one, [r_ for r_ in rows if r_["split"] == "val"]))
    ms = np.array([r[0] for r in res]); used = Counter(r[1]["used"] for r in res); reasons = Counter(r[1]["reason"] for r in res if r[1]["reason"])
    scores = [r[1]["score"] for r in res]; labels = [r[2] for r in res]
    print(f"val calls: {len(ms)} (parallel={PAR})  wall latency p50={np.percentile(ms,50):.0f} ms  p95={np.percentile(ms,95):.0f} ms  max={ms.max():.0f} ms")
    print(f"path used: {dict(used)}  reasons: {dict(reasons)}   AUC of returned scores: {roc_auc_score(labels, scores):.3f}")
    assert np.percentile(ms, 95) < 3000
    print("OK")
finally:
    srv.terminate()
