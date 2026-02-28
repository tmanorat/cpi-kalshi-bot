# ──────────────────────────────────────────────────────────────────
# CPI Kalshi Bot — Dockerfile
# Base: python:3.11-slim
# Both "dashboard" and "bot" services use this same image.
# ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Stream logs immediately; make /app the Python root
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# libgomp1 is a runtime dep of xgboost (OpenMP threading)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (own layer so rebuilds skip pip on code-only changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "openpyxl>=3.1.0"

# ── Application source ──────────────────────────────────────────────
# .dockerignore excludes: venv/, __pycache__/, *.pyc, .env, *.db,
#   models/cpi_forecaster.pkl, logs/
COPY . .

# Create directories that will be bind-mounted as named volumes.
# Docker will overlay the empty volume onto these paths at runtime;
# the directories must pre-exist in the image for the mount to work.
RUN mkdir -p /app/logs /app/models /app/data

# ── Entrypoint: initialise DB tables, then exec the CMD ────────────
# init_db() is idempotent (create_all + portfolio seed guard).
# Running it in every container ensures tables exist regardless of
# start order.
RUN printf '#!/bin/bash\nset -e\npython -c "from data.database import init_db; init_db()"\nexec "$@"\n' \
        > /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
