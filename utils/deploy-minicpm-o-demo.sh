#!/usr/bin/env bash
set -euo pipefail

# Quick-and-dirty MiniCPM-o-Demo deploy for testing: runs worker.py +
# gateway.py directly on the host (no containers), installing into the
# currently active (uv) venv rather than creating a new one. Assumes an
# NVIDIA GPU + drivers are already present, and that a venv is already
# active. Run from a directory containing both ./MiniCPM-o-Demo/ and
# ./vllm-omni-aux/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_HOST_PATH="$(hf download openbmb/MiniCPM-o-4_5)"

cd MiniCPM-o-Demo

uv pip install "torch==2.8.0" "torchaudio==2.8.0"
uv pip install -r requirements.txt

pkill -f "gateway.py|worker.py" || true

mkdir -p certs tmp
if [[ ! -f certs/cert.pem ]]; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
        -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"
fi

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nohup python worker.py \
    --model-path "$MODEL_HOST_PATH" --worker-index 0 --gpu-id 0 \
    > tmp/worker_0.log 2>&1 &
WORKER_PID=$!
disown

echo "Waiting for worker to become healthy..."
"$SCRIPT_DIR/poll-server-health.sh" "$WORKER_PID" http://127.0.0.1:22400/health tmp/worker_0.log

PYTHONPATH=. nohup python gateway.py \
    --port 8006 --internal-port 8007 \
    > tmp/gateway.log 2>&1 &
GATEWAY_PID=$!
disown

echo "Waiting for gateway to become healthy..."
"$SCRIPT_DIR/poll-server-health.sh" "$GATEWAY_PID" https://127.0.0.1:8006/health tmp/gateway.log

curl -X PUT http://127.0.0.1:8007/internal/workers/local-worker \
    -H 'content-type: application/json' \
    --data '{"endpoint":"127.0.0.1:22400","gpu_group":"gpu-0"}'

echo "Gateway ready at: https://$(hostname -I | awk '{print $1}'):8006/"
echo "Worker log:  $PWD/tmp/worker_0.log"
echo "Gateway log: $PWD/tmp/gateway.log"
