#!/usr/bin/env bash
set -euo pipefail

# Run MiniCPM-o e2e tests (real server startup + inference) against a
# vllm-omni checkout at /app/vllm-omni.

REPO_DIR="/app/vllm-omni"
VENV_PYTHON="/app/venv/bin/python3"

cd "$REPO_DIR"

# Target explicit MiniCPM-o e2e test paths: this excludes standalone demo
# driver scripts (tests/e2e/online_serving/run_*.py) and non-test helper
# modules (tests/e2e/online_serving/helpers/*.py), which aren't pytest tests.
mapfile -t TEST_PATHS < <(
    find tests/e2e -iname "*minicpm*" \
        \( -path "*/helpers/*" -o -name "run_*.py" \) -prune -o \
        -iname "*minicpm*" -print
)

timeout 3600 "$VENV_PYTHON" -m pytest "${TEST_PATHS[@]}" -v
