#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# update.sh — Fast update: rsync then restart WITHOUT rebuilding
#
# Usage:  ./update.sh ROOT@DROPLET_IP
#
# Use this when you only changed .py files.
# If you changed requirements.txt or the Dockerfile, use deploy.sh.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

TARGET="${1:?Usage: ./update.sh ROOT@DROPLET_IP}"
REMOTE_DIR="/opt/cpi-kalshi-bot"

# ── Pre-flight ─────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "ERROR: .env not found locally."
    exit 1
fi

# ── Sync changed files only ───────────────────────────────────────
echo "==> Syncing changed files to ${TARGET}:${REMOTE_DIR} ..."
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

echo "==> Copying .env ..."
rsync -az .env "${TARGET}:${REMOTE_DIR}/.env"

# ── Restart without rebuild ───────────────────────────────────────
echo "==> Restarting containers (no --build) ..."
ssh "${TARGET}" "cd ${REMOTE_DIR} && docker compose down && docker compose up -d"

# ── Done ──────────────────────────────────────────────────────────
DROPLET_IP="$(echo "${TARGET}" | cut -d@ -f2)"
echo ""
echo "✅ Updated!"
echo "   Dashboard → http://${DROPLET_IP}:8001"
