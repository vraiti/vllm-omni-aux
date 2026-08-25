#!/usr/bin/env bash
set -euo pipefail

# Run a single MiniCPM-o offline-inference test with synchronous CUDA
# execution so a crash during sampling produces a catchable Python
# traceback at the actual failing kernel launch, instead of silently
# killing the process asynchronously.

REPO_DIR="/app/vllm-omni"
VENV_PYTHON="/app/venv/bin/python3"

cd "$REPO_DIR"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

timeout 600 "$VENV_PYTHON" -u -m pytest \
    tests/e2e/offline_inference/test_minicpmo_4_5.py::test_text_to_text \
    -v -s --tb=long
