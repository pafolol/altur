FROM python:3.11.13-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements-runtime.lock ./
RUN pip install --no-cache-dir -r requirements-runtime.lock && useradd --uid 10001 --create-home app
COPY behavior/ ./behavior/
COPY artifacts/models/behavior.json ./artifacts/models/behavior.json
COPY artifacts/vad/silero_vad.onnx ./artifacts/vad/silero_vad.onnx
COPY THIRD_PARTY_NOTICES.md ./
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "behavior.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "4", "--no-access-log"]
