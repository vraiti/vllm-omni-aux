#!/usr/bin/env bash
set -euo pipefail

# Deploys the MiniCPM-o-Demo server stack (gateway + worker-backend) via
# Docker Compose on this host, following the "Docker Deployment
# (Recommended)" path from MiniCPM-o-Demo/README.md.
#
# Assumes:
#   - an NVIDIA GPU is present with drivers already installed (nvidia-smi works)
#   - docker, the Compose v2 plugin, and the NVIDIA Container Toolkit are
#     already installed
# This script does not install any of the above -- it fails fast with
# guidance if a prerequisite is missing, rather than silently trying to
# install system packages on an unknown host.
#
# Weights: `hf download openbmb/MiniCPM-o-4_5` pre-fetches the model into
# the host's HF cache (~/.cache/huggingface) before bringing up containers.
# MODEL_HOST_PATH (the directory bind-mounted read-only into worker-backend
# at /models/MiniCPM-o-4_5) defaults to the resulting cache snapshot, and
# the whole ~/.cache is additionally mounted read-only into each
# worker-backend container (see minicpm-o-hf-cache.override.yml) so any
# repo-id-based huggingface_hub/transformers lookups also resolve locally.
#
# Only brings up as many worker-backend services as there are GPUs
# (docker-compose.yml as checked in defines worker-backend-0/1; for >2 GPUs
# edit docker-compose.yml per the README and re-run).
#
# Usage (run from inside the MiniCPM-o-Demo checkout, or pass its path):
#   ./deploy-minicpm-o-demo.sh [demo_repo_dir]
#
# Env vars:
#   MODEL_HOST_PATH    host dir containing the MiniCPM-o-4_5 weights
#                      (default: resolved from `hf download`)
#   HF_CACHE_HOST_PATH host HF cache dir mounted into the container
#                      (default: $HOME/.cache)
#   MINICPM_DEMO_DIR   MiniCPM-o-Demo checkout (default: $PWD, or $1 if given)
#   GATEWAY_HOST_PORT  gateway port on this host (default: 8006)

DEMO_DIR="${1:-${MINICPM_DEMO_DIR:-$PWD}}"
GATEWAY_HOST_PORT="${GATEWAY_HOST_PORT:-8006}"
HF_CACHE_HOST_PATH="${HF_CACHE_HOST_PATH:-$HOME/.cache}"

if [[ ! -d "$DEMO_DIR" ]]; then
    echo "ERROR: MiniCPM-o-Demo checkout not found at $DEMO_DIR (pass it as \$1 or set MINICPM_DEMO_DIR)" >&2
    exit 1
fi

command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found -- NVIDIA driver is not installed" >&2; exit 1; }
nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi failed -- GPU/driver not usable" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: docker compose v2 plugin not found" >&2; exit 1; }
docker info 2>/dev/null | grep -qi nvidia || echo "WARNING: NVIDIA Container Toolkit runtime not detected in 'docker info' -- GPU containers may fail to start" >&2

if [[ -z "${MODEL_HOST_PATH:-}" ]]; then
    command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' CLI not found -- pip install -U huggingface_hub" >&2; exit 1; }
    echo "Ensuring openbmb/MiniCPM-o-4_5 weights are present in $HF_CACHE_HOST_PATH ..."
    HF_HOME="$HF_CACHE_HOST_PATH/huggingface" hf download openbmb/MiniCPM-o-4_5
    MODEL_HOST_PATH=$(HF_HOME="$HF_CACHE_HOST_PATH/huggingface" python3 -c \
        "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/MiniCPM-o-4_5', local_files_only=True))")
fi
if [[ ! -d "$MODEL_HOST_PATH" ]]; then
    echo "ERROR: MODEL_HOST_PATH ($MODEL_HOST_PATH) does not exist" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERRIDE_FILE="$SCRIPT_DIR/minicpm-o-hf-cache.override.yml"
COMPOSE=(docker compose -f docker-compose.yml -f "$OVERRIDE_FILE")

cd "$DEMO_DIR"

mkdir -p certs data
if [[ ! -f certs/cert.pem || ! -f certs/key.pem ]]; then
    echo "Generating self-signed TLS cert (browser mic/camera access requires https)..."
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
        -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"
fi

GPU_COUNT=$(nvidia-smi -L | wc -l)
SERVICES=(gateway)
for ((i = 0; i < GPU_COUNT && i < 2; i++)); do
    SERVICES+=("worker-backend-$i")
done
if (( GPU_COUNT > 2 )); then
    echo "WARNING: $GPU_COUNT GPUs detected but docker-compose.yml only defines worker-backend-0/1; only 2 will be used. Edit docker-compose.yml to use more." >&2
fi

echo "Starting: ${SERVICES[*]} (GPU_COUNT=$GPU_COUNT)"
MODEL_HOST_PATH="$MODEL_HOST_PATH" GATEWAY_HOST_PORT="$GATEWAY_HOST_PORT" HF_CACHE_HOST_PATH="$HF_CACHE_HOST_PATH" \
    "${COMPOSE[@]}" up -d --build "${SERVICES[@]}"

echo "Waiting for worker-backend-0 to become healthy (model load can take several minutes)..."
for _ in $(seq 1 90); do
    status=$("${COMPOSE[@]}" ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep '^minicpm-wb-0 ' || true)
    if grep -qi '(healthy)' <<<"$status"; then
        HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
        echo "worker-backend-0 is healthy."
        echo "Gateway ready at: https://${HOST_IP:-<this-host-ip>}:${GATEWAY_HOST_PORT}/"
        exit 0
    fi
    sleep 10
done

echo "WARNING: worker-backend-0 did not report healthy within 15 minutes; check '${COMPOSE[*]} logs -f worker-backend-0'" >&2
exit 1
