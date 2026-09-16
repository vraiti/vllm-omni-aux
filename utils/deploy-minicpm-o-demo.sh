#!/usr/bin/env bash
set -euo pipefail

# Quick-and-dirty MiniCPM-o-Demo deploy for testing. Assumes an NVIDIA GPU +
# drivers are already present, and that podman + podman-compose are
# installed. Run from a directory containing both ./MiniCPM-o-Demo/ and
# ./vllm-omni-aux/.

MODEL_HOST_PATH="$(hf download openbmb/MiniCPM-o-4_5)"

cd MiniCPM-o-Demo

mkdir -p certs data
if [[ ! -f certs/cert.pem ]]; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
        -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"
fi

MODEL_HOST_PATH="$MODEL_HOST_PATH" podman compose up -d --build
