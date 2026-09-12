"""Score one WAV against a running server:  python detect.py samples/call.wav [http://127.0.0.1:8000]"""
import base64, json, sys, urllib.request

path, url = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000")
body = json.dumps({"audio": base64.b64encode(open(path, "rb").read()).decode()}).encode()
req = urllib.request.Request(f"{url}/detect", body, {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print(json.dumps(json.load(r), indent=2, ensure_ascii=False))
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()}"); sys.exit(1)
