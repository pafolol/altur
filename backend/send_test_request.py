import base64
import json
import sys
import time

import requests


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python send_test_request.py ruta/al/audio.wav")
    with open(sys.argv[1], "rb") as audio_file:
        payload = {"audio": base64.b64encode(audio_file.read()).decode("ascii")}
    print("Sending request...\n")
    started = time.perf_counter()
    try:
        response = requests.post("http://127.0.0.1:8000/detect", json=payload, timeout=120)
        print(f"Status: {response.status_code}")
        print(f"Response time: {time.perf_counter() - started:.3f} s\n")
        try:
            print(json.dumps(response.json(), indent=4, ensure_ascii=True))
        except ValueError:
            print(response.text)
    except requests.RequestException as exc:
        raise SystemExit(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
