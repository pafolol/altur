"""Fase 0 acceptance: manifest has 353 calls and every audio + turns file exists."""
import csv, pathlib, sys
from collections import Counter

ROOT = pathlib.Path(__file__).parent / "data" / "hackmty26"
rows = list(csv.DictReader(open(ROOT / "manifest.csv")))
missing_audio = [r["anon_id"] for r in rows if not (ROOT / "audio" / f"{r['anon_id']}.wav").exists()]
missing_turns = [r["anon_id"] for r in rows if not (ROOT / "turns" / f"{r['anon_id']}.json").exists()]

print(f"manifest rows : {len(rows)}")
print(f"audio present : {len(rows) - len(missing_audio)}  (missing {len(missing_audio)})")
print(f"turns present : {len(rows) - len(missing_turns)}  (missing {len(missing_turns)})")
for (split, label), n in sorted(Counter((r["split"], r["label"]) for r in rows).items()):
    print(f"  {split:5s} {label:9s} {n}")
assert len(rows) == 353, len(rows)
assert not missing_audio and not missing_turns, (missing_audio[:3], missing_turns[:3])
print("OK")
