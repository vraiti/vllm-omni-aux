#!/usr/bin/env bash
# Run pytest against the synced vllm-omni checkout, inside its venv.
# Invoke via run-remote-rsync.sh so the tree is synced first, e.g.:
#   run-remote-rsync.sh vllm-omni run-pytest.sh entrypoints/openai_api/test_realtime_session.py -m "core_model and cpu"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
VLLM_OMNI_DIR="$PROJECT_ROOT/vllm-omni"

if ! "$VENV_DIR/bin/python3" -c "import pytest, pytest_mock" 2>/dev/null; then
    echo "Installing pytest/pytest-mock into $VENV_DIR ..."
    "$HOME/.local/bin/uv" pip install --python "$VENV_DIR/bin/python3" pytest pytest-mock
fi

cd "$VLLM_OMNI_DIR/tests"
exec "$VENV_DIR/bin/python3" -m pytest "$@"
