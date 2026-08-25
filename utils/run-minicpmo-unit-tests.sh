#!/usr/bin/env bash
set -euo pipefail

# Run MiniCPM-o unit tests against a vllm-omni checkout at /app/vllm-omni.
# Excludes e2e/, examples/, and dfx/perf/, which spin up full servers or
# run performance benchmarks rather than unit-level checks.

REPO_DIR="/app/vllm-omni"
VENV_PYTHON="/app/venv/bin/python3"

cd "$REPO_DIR"

# Target only MiniCPM-o test paths directly instead of collecting the whole
# suite with -k: collecting everything pulls in unrelated modules (e.g.
# diffusion/hunyuan_image3) whose import errors abort the entire pytest
# session before any MiniCPM-o test gets to run.
mapfile -t TEST_PATHS < <(
    find tests -iname "*minicpm*" \
        \( -path "*/e2e/*" -o -path "*/examples/*" -o -path "*/dfx/perf/*" -o -path "tests/assets/*" \) -prune -o \
        -iname "*minicpm*" -print
)

timeout 1800 "$VENV_PYTHON" -m pytest "${TEST_PATHS[@]}" -v
