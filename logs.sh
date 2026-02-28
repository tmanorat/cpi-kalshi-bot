#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# logs.sh — Tail live logs from running containers
#
# Usage:  ./logs.sh ROOT@DROPLET_IP [dashboard|bot|all]
#
# Defaults to "all" when no service is specified.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

TARGET="${1:?Usage: ./logs.sh ROOT@DROPLET_IP [dashboard|bot|all]}"
SERVICE="${2:-all}"
REMOTE_DIR="/opt/cpi-kalshi-bot"

if [ "${SERVICE}" = "all" ]; then
    ssh "${TARGET}" "cd ${REMOTE_DIR} && docker compose logs --tail=100 -f"
else
    ssh "${TARGET}" "cd ${REMOTE_DIR} && docker compose logs --tail=100 -f ${SERVICE}"
fi
