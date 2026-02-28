#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# deploy.sh — Full deploy: rsync + docker compose up --build
#
# Usage:  ./deploy.sh ROOT@DROPLET_IP
#
# Run this whenever you change requirements.txt or the Dockerfile.
# For Python-only changes use update.sh (faster, no rebuild).
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

TARGET="${1:?Usage: ./deploy.sh ROOT@DROPLET_IP}"
REMOTE_DIR="/opt/cpi-kalshi-bot"

# ── Pre-flight ─────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "ERROR: .env not found locally. Create it before deploying."
    exit 1
fi

# ── Ensure remote directory exists ────────────────────────────────
echo "==> Ensuring ${REMOTE_DIR} exists on ${TARGET} ..."
ssh "${TARGET}" "mkdir -p ${REMOTE_DIR}"

# ── Sync project files ────────────────────────────────────────────
echo "==> Syncing files to ${TARGET}:${REMOTE_DIR} ..."
rsync -avz --progress \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.db' \
    --exclude='*.db-shm' \
    --exclude='*.db-wal' \
    --exclude='models/cpi_forecaster.pkl' \
    --exclude='logs/' \
    --exclude='.env' \
    ./ "${TARGET}:${REMOTE_DIR}/"

# ── Copy .env separately (never let rsync log the path in verbose output) ──
echo "==> Copying .env ..."
rsync -az .env "${TARGET}:${REMOTE_DIR}/.env"

# ── Build and start ───────────────────────────────────────────────
echo "==> Running docker compose down && up --build on server ..."
ssh "${TARGET}" "cd ${REMOTE_DIR} && docker compose down && docker compose up -d --build"

# Give containers a moment to finish the entrypoint (init_db) and settle
echo "    Waiting for containers to be ready ..."
sleep 5

# ── Upload trained model if it exists locally ─────────────────────
# The pkl is excluded from rsync so it must be pushed separately.
# scp stages it on the host, then docker cp loads it into the
# models_data named volume via the running container.
if [ -f models/cpi_forecaster.pkl ]; then
    echo "==> Uploading models/cpi_forecaster.pkl to ${TARGET} ..."
    scp models/cpi_forecaster.pkl "${TARGET}:${REMOTE_DIR}/models/cpi_forecaster.pkl"
    echo "==> Copying model into models_data Docker volume ..."
    ssh "${TARGET}" "docker cp ${REMOTE_DIR}/models/cpi_forecaster.pkl cpi-bot:/app/models/cpi_forecaster.pkl"
    echo "    Model installed into volume."
else
    echo "    No local models/cpi_forecaster.pkl — will check if server needs training."
fi

# ── Train on server if model is still missing from the volume ─────
# Only runs on first deploy (or if the volume was wiped).
# Uses the bot container since it already has the full engine loaded.
echo "==> Checking for trained model inside container ..."
if ssh "${TARGET}" "docker exec cpi-bot test -f /app/models/cpi_forecaster.pkl" 2>/dev/null; then
    echo "    Model present in volume. Skipping training."
else
    echo "==> Model not found — running python engine.py train inside container ..."
    echo "    (This takes ~3 minutes. Logs stream below.)"
    ssh "${TARGET}" "cd ${REMOTE_DIR} && docker compose exec -T bot python engine.py train"
    echo "    Training complete."
fi

# ── Done ──────────────────────────────────────────────────────────
DROPLET_IP="$(echo "${TARGET}" | cut -d@ -f2)"
echo ""
echo "✅ Deployed successfully!"
echo "   Dashboard → http://${DROPLET_IP}:8001"
echo "   API docs  → http://${DROPLET_IP}:8001/docs"
