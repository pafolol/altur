"""
THE CALL LOG - a small SQLite store of every verdict the endpoint has produced.

The detector itself is stateless: one WAV in, one verdict out. The admin panel is not - it asks for the
last 24 hours of calls, a latency series and an uptime figure, and none of those can be answered by a
process that remembers nothing. This is that memory, and it is deliberately the smallest thing that
answers those three questions honestly.

    record(verdict)        append one decision, with every layer's own score, and mirror it to the Tiger
                           telemetry service (tiger-telemetry/) when TELEMETRY_URL is set
    recent(hours)          the calls the panel lists
    latency_series(...)    p50 per bucket, oldest first
    health_stats(hours)    uptime and volume over a window

SQLite because it needs no server, no container and no credentials: one file under outputs/, created on
first use. WAL mode so a read never blocks the endpoint. Nothing here is on the hot path of a verdict -
record() is called after the response is computed, and a failure to write is logged and swallowed rather
than turned into a 500: losing a log row is not worth failing a detection over.

NO AUDIO IS STORED. Only the scores, the verdict and the timings. The audio arrives, is scored, and is
gone - the dataset terms say not to redistribute the recordings, and a database of them is exactly that.
"""
import json
import os
import sqlite3
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import config

DB_PATH = config.OUTPUTS_DIR / "fusion" / "calls.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id            TEXT PRIMARY KEY,
    at            TEXT NOT NULL,           -- ISO-8601 UTC, when the verdict was produced
    duration_s    REAL,                    -- length of the audio, not of the request
    channels      INTEGER,
    verdict       TEXT NOT NULL,           -- human | synthetic | abstained
    confidence    REAL NOT NULL,
    probability   REAL NOT NULL,           -- the fused P(synthetic)
    latency_ms    REAL NOT NULL,
    decided_at_s  REAL,                    -- when the running acoustic score first settled; NULL if never
    queue         TEXT NOT NULL DEFAULT 'api',
    trap          TEXT NOT NULL DEFAULT 'none',
    layers        TEXT NOT NULL,           -- JSON: per-layer score, quality, abstained, consulted, role
    ok            INTEGER NOT NULL DEFAULT 1   -- 0 = the request failed; used for uptime
);
CREATE INDEX IF NOT EXISTS calls_at ON calls (at DESC);
"""

_lock = threading.Lock()
_conn = None


def connect():
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")       # a reader never blocks the endpoint
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(SCHEMA)
        _conn.commit()
    return _conn


def _now():
    return datetime.now(timezone.utc)


def decided_at(layer_details, threshold=0.5):
    """
    When the call stopped being ambiguous, in seconds from its start.

    The acoustic layer scores 4 s chunks of caller speech and takes the mean of their log-odds, so the
    running mean after k chunks is exactly what it would have answered having heard only those k. This
    walks that forward and returns the end of the first chunk after which the running verdict never
    changes again - a real "it was decided by here", not a guess.

    None when the layer abstained, produced no chunks, or never settles.
    """
    acoustic = next((l for l in layer_details if l.get("key") == "acoustic"), None)
    if not acoustic or acoustic.get("abstained") or not acoustic.get("scored", True):
        return None
    d = acoustic.get("details") or {}
    scores, spans = d.get("chunk_scores") or [], d.get("chunk_spans") or []
    if not scores or len(spans) != len(scores):
        return None
    final = sum(scores) / len(scores) >= 0
    running, settled_from = 0.0, None
    for i, s in enumerate(scores):
        running += s
        if ((running / (i + 1)) >= 0) == final:
            if settled_from is None:
                settled_from = i
        else:
            settled_from = None                      # it flipped back: not settled yet
    if settled_from is None:
        return None
    return round(float(spans[settled_from][1]), 2)   # the end of the chunk it settled on


def _merge_layer_views(results, contributions):
    """LayerResult dicts + combine() contributions, joined on the layer key. Either may be empty."""
    by_key = {c["key"]: c for c in contributions if isinstance(c, dict) and "key" in c}
    out = []
    for r in results or contributions:
        if not isinstance(r, dict) or "key" not in r:
            continue
        out.append(dict(by_key.get(r["key"], {}), **r) if results else dict(r))
    return out


def _layer_score(layers, key):
    """One layer's P(synthetic), or None when it abstained or was never asked."""
    l = next((l for l in layers if l.get("key") == key), None)
    if not l or l.get("abstained") or l.get("scored") is False or l.get("probability") is None:
        return None
    return round(float(l["probability"]), 4)


def mirror_to_telemetry(call_id, verdict, probability, layers, latency_ms):
    """
    Send the same decision to the Tiger telemetry service (tiger-telemetry/: a separate process with its own
    PostgreSQL) when TELEMETRY_URL is set. Off the hot path: a daemon thread, a 3 s timeout, and a failure
    is one printed line. The SQLite log is the source of truth; this is the copy that leaves the machine.
    """
    url = os.environ.get("TELEMETRY_URL", "").rstrip("/")
    if not url:
        return
    body = json.dumps({
        "call_id": call_id,
        "is_synthetic": bool(verdict.get("is_synthetic")),
        "confidence": min(max(float(verdict.get("confidence", 0.5)), 0.0), 1.0),
        "acoustic_score": _layer_score(layers, "acoustic"),
        "behavioral_score": _layer_score(layers, "behaviour"),
        "semantic_score": _layer_score(layers, "semantic"),
        "final_score": min(max(float(probability), 0.0), 1.0),
        "latency_ms": max(float(latency_ms), 0.0),
    }).encode()

    def post():
        req = urllib.request.Request(url + "/telemetry", body, {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=3).close()
        except Exception as exc:                      # noqa: BLE001 - a copy is not worth a stack trace
            print(f"[store] telemetry not mirrored: {type(exc).__name__}: {exc}")

    threading.Thread(target=post, daemon=True).start()


def record(verdict, duration_s=None, channels=None, queue="api", trap="none", ok=True, call_id=None):
    """
    Append one decision. Never raises: a log row is not worth failing a detection over.

    `verdict` is the dict score_call() returns, so the caller does not have to reshape anything.
    """
    try:
        details = verdict.get("details") or {}
        # Two views of the same layers: details["layers"] are the LayerResults (probability, quality,
        # abstained, scored, reason) and verdict["layers"] are combine()'s contributions (role, share,
        # consulted, weight). The log wants both, so they are merged by key.
        layers = _merge_layer_views(details.get("layers") or [], verdict.get("layers") or [])
        p = float(details.get("synthetic_probability", verdict.get("synthetic_probability", 0.5)))
        decisive = verdict.get("decisive", True)
        call_id = call_id or uuid.uuid4().hex[:12]
        latency_ms = float(details.get("latency_ms", 0))
        row = (
            call_id,
            _now().isoformat(),
            duration_s if duration_s is not None else details.get("duration_s"),
            channels,
            "abstained" if not decisive else ("synthetic" if verdict.get("is_synthetic") else "human"),
            float(verdict.get("confidence", 0.5)),
            p,
            latency_ms,
            decided_at(layers),
            queue, trap,
            json.dumps([{k: l.get(k) for k in
                         ("key", "display", "probability", "quality", "abstained", "scored",
                          "role", "consulted", "share", "reason")}
                        for l in layers]),
            1 if ok else 0,
        )
        if ok:                                        # a failed request is uptime, not a detection
            mirror_to_telemetry(call_id, verdict, p, layers, latency_ms)
        with _lock:
            c = connect()
            c.execute("INSERT INTO calls (id, at, duration_s, channels, verdict, confidence, probability,"
                      " latency_ms, decided_at_s, queue, trap, layers, ok) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            c.commit()
        return row[0]
    except Exception as exc:                          # noqa: BLE001 - logging must never break scoring
        print(f"[store] could not record the call: {type(exc).__name__}: {exc}")
        return None


def _since(hours):
    return (_now() - timedelta(hours=hours)).isoformat()


def recent(hours=24, limit=500):
    with _lock:
        rows = connect().execute(
            "SELECT * FROM calls WHERE at >= ? AND ok = 1 ORDER BY at DESC LIMIT ?",
            (_since(hours), limit)).fetchall()
    return [dict(r) | {"layers": json.loads(r["layers"])} for r in rows]


def latency_series(hours=24, buckets=48):
    """p50 latency per bucket, oldest first - the shape the panel's sparkline wants."""
    with _lock:
        rows = connect().execute("SELECT at, latency_ms FROM calls WHERE at >= ? AND ok = 1",
                                 (_since(hours),)).fetchall()
    if not rows:
        return [0] * buckets
    end = _now()
    width = timedelta(hours=hours) / buckets
    binned = [[] for _ in range(buckets)]
    for r in rows:
        try:
            age = end - datetime.fromisoformat(r["at"])
        except ValueError:
            continue
        i = buckets - 1 - int(age / width)
        if 0 <= i < buckets:
            binned[i].append(r["latency_ms"])
    out, last = [], 0
    for b in binned:
        if b:
            b.sort()
            last = round(b[len(b) // 2])
        out.append(last)                              # carry the last known value through empty buckets
    return out


def health_stats(hours=24):
    with _lock:
        row = connect().execute(
            "SELECT COUNT(*) n, SUM(ok) ok FROM calls WHERE at >= ?", (_since(hours),)).fetchone()
    n, ok = int(row["n"] or 0), int(row["ok"] or 0)
    return {"n": n, "ok": ok, "uptime_24h": round(100.0 * ok / n, 2) if n else 100.0}


def prune(days=30):
    """Keep the file small; nothing here is meant to be a warehouse."""
    with _lock:
        c = connect()
        c.execute("DELETE FROM calls WHERE at < ?", ((_now() - timedelta(days=days)).isoformat(),))
        c.commit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect the call log.")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args()
    if args.prune:
        prune()
    rows = recent(args.hours)
    h = health_stats(args.hours)
    print(f"{DB_PATH}\n{h['n']} calls in the last {args.hours} h, {h['uptime_24h']:.2f}% ok\n")
    print(f"{'id':14s} {'at':22s} {'verdict':10s} {'conf':>5s} {'ms':>6s} {'decided':>8s}  layers")
    for r in rows[:25]:
        d = "-" if r["decided_at_s"] is None else f"{r['decided_at_s']:.1f}s"
        line = " ".join(f"{l['key'][:4]}={'--' if l.get('abstained') or l.get('scored') is False else format(l['probability'], '.2f')}"
                        for l in r["layers"])
        print(f"{r['id']:14s} {r['at'][:19]:22s} {r['verdict']:10s} {r['confidence']:5.2f} "
              f"{r['latency_ms']:6.0f} {d:>8s}  {line}")


if __name__ == "__main__":
    main()
