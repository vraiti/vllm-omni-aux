#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:?Usage: $0 <vllm-omni-repo-path>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/../venv"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
fi

export FLUX2_DEBUG_DIR=/tmp/flux2_debug
export HF_TOKEN=$(cat ~/.secret/hf)

TP_SIZE=$(nvidia-smi -L | wc -l)

cd "$REPO_DIR"

python examples/offline_inference/image_to_image/image_edit.py \
    --model black-forest-labs/FLUX.2-dev \
    --image "$SCRIPT_DIR/sample-input.png" \
    --prompt "replace the bunny in the image with dog." \
    --output outputs/flux2-dev-edit.png \
    --seed 42 \
    --tensor-parallel-size "$TP_SIZE" \
    --num-inference-steps 50 \
    --guidance-scale 4.0
