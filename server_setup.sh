#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# server_setup.sh — One-time setup for a fresh Ubuntu 22.04 droplet
#
# Usage:  ./server_setup.sh ROOT@DROPLET_IP
#
# What it does:
#   • Installs Docker CE + the docker compose v2 plugin (if absent)
#   • Creates /opt/cpi-kalshi-bot/
#   • Does NOT touch any other directory or Docker project on the server
#
# Safe to re-run — all steps are idempotent.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

TARGET="${1:?Usage: ./server_setup.sh ROOT@DROPLET_IP}"

echo "==> Configuring server ${TARGET} ..."

ssh "${TARGET}" 'bash -s' << 'REMOTE'
set -euo pipefail

# ── Docker CE ─────────────────────────────────────────────────────
if command -v docker &>/dev/null; then
    echo "Docker already installed: $(docker --version)"
else
    echo "Installing Docker CE ..."
    apt-get update -q
    apt-get install -y -q ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -q
    apt-get install -y -q \
        docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
    echo "Docker installed: $(docker --version)"
fi

# ── Docker Compose v2 plugin ──────────────────────────────────────
if docker compose version &>/dev/null; then
    echo "Docker Compose available: $(docker compose version)"
else
    echo "ERROR: docker compose plugin missing after install."
    exit 1
fi

# ── Project directory ─────────────────────────────────────────────
mkdir -p /opt/cpi-kalshi-bot
echo "Project directory ready: /opt/cpi-kalshi-bot"

echo ""
echo "✅ Server setup complete."
echo "   No existing Docker projects were modified."
echo "   Existing containers/networks/volumes are untouched."
REMOTE

echo "✅ Done. Next step:  ./deploy.sh ${TARGET}"
