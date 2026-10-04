# Build from the repo root:  docker build -f docker/Dockerfile --target serving -t pmaint-api .
FROM python:3.12-slim AS base

# libgomp: OpenMP runtime required by the xgboost wheel
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src PMAINT_HOME=/app MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app

COPY docker/requirements-serving.txt docker/requirements-serving.txt
RUN pip install -r docker/requirements-serving.txt

COPY src ./src
RUN useradd --create-home app && mkdir -p /app/logs /app/reports && chown -R app /app
USER app


# ---- API image ------------------------------------------------------------------------------
FROM base AS serving
ENV PMAINT_PRED_LOG=/app/logs/predictions.jsonl
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "pmaint.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]


# ---- Drift monitor job (adds evidently; needs data/raw mounted for the reference set) ---------
FROM base AS monitor
USER root
COPY docker/requirements-monitor.txt docker/requirements-monitor.txt
RUN pip install -r docker/requirements-monitor.txt
USER app
CMD ["python", "-m", "pmaint.monitoring.drift", "--current", "/app/logs/predictions.jsonl", "--html", "/app/reports/drift.html"]
