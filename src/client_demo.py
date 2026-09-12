"""
Sends a call to a running /detect endpoint exactly the way the judges do (base64 WAV in JSON).

Run:  python src/client_demo.py path/to/call.wav [--url http://localhost:8000/detect]
      python src/client_demo.py --val 5      (5 random validation calls with their labels)
"""
import argparse
import json
import time
import urllib.request

import _bootstrap  # noqa: F401
import audio
import config


def send(url, path):
    payload = json.dumps({"audio": audio.encode_wav_base64(path)}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--url", default="http://localhost:8000/detect")
    parser.add_argument("--val", type=int, default=0, help="also send N random validation calls")
    args = parser.parse_args()
    items = [(p, None) for p in args.paths]
    if args.val:
        import dataset
        df = dataset.load_split("val").sample(n=args.val, random_state=config.SEED)
        items += [(r.path, r.label) for r in df.itertuples()]
    for path, truth in items:
        body, dt = send(args.url, path)
        line = f"{path}: {json.dumps({k: v for k, v in body.items() if k != 'details'})}  ({dt * 1000:.0f} ms round trip)"
        if truth:
            ok = body.get("is_synthetic") == (truth == "synthetic")
            line += f"   truth={truth} -> {'ok' if ok else 'WRONG'}"
        print(line)


if __name__ == "__main__":
    main()
