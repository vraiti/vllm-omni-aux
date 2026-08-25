#!/usr/bin/env bash
set -euo pipefail

# Run MiniCPM-o unit tests against a vllm-omni checkout at /app/vllm-omni.
# Excludes e2e/, examples/, and dfx/perf/, which spin up full servers or
# run performance benchmarks rather than unit-level checks.

REPO_DIR="/app/vllm-omni"
VENV_PYTHON="/app/venv/bin/python3"

cd "$REPO_DIR"

"$VENV_PYTHON" -m pytest tests \
    -k "minicpmo or minicpm" \
    --ignore=tests/e2e \
    --ignore=tests/examples \
    --ignore=tests/dfx/perf \
    -v
