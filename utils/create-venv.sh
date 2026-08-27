#!/usr/bin/env bash
set -euo pipefail

# Assumes vllm-omni is already present under /app (e.g. synced by
# run-remote-rsync.sh) -- this script only builds the venv and installs into
# it, it does not clone anything.
VLLM_OMNI_DIR="/app/vllm-omni"

# With no args, create the venv at /app/venv. With args, pass them straight
# through to `uv venv` (e.g. `create-venv.sh other-venv --python
# path/to/other/python`) -- the first non-flag argument is then the venv
# path, same as uv venv's own convention, defaulting to uv's own default
# (.venv in cwd) if only flags were given.
if [[ $# -eq 0 ]]; then
    VENV_ARGS=(/app/venv)
    VENV_DIR="/app/venv"
elif [[ "$1" != -* ]]; then
    VENV_ARGS=("$@")
    VENV_DIR="$1"
else
    VENV_ARGS=("$@")
    VENV_DIR=".venv"
fi

echo "Creating venv..."
uv venv "${VENV_ARGS[@]}"
source "$VENV_DIR/bin/activate"

VLLM_VERSION=$(grep -oP 'VLLM_VERSION[= ]+v?\K[0-9]+\.[0-9]+\.[0-9]+' "$VLLM_OMNI_DIR"/docker/Dockerfile.xpu | head -1)
if [[ -z "$VLLM_VERSION" ]]; then
    echo "ERROR: could not determine VLLM_VERSION from docker/Dockerfile.xpu" >&2
    exit 1
fi

echo "Installing vllm==$VLLM_VERSION..."
uv pip install "vllm==$VLLM_VERSION"

FLASHINFER_VERSION=$(uv pip show flashinfer-python | grep -oP '^Version: \K.*')
if [[ -z "$FLASHINFER_VERSION" ]]; then
    echo "ERROR: flashinfer not found after vllm install" >&2
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_TAG="cu${CUDA_MAJOR}0"
echo "Installing flashinfer-jit-cache==$FLASHINFER_VERSION ($CUDA_TAG)..."
if ! uv pip install "flashinfer-jit-cache==$FLASHINFER_VERSION" \
    --index-url "https://flashinfer.ai/whl/${CUDA_TAG}" 2>/dev/null; then
    echo "flashinfer-jit-cache not available, skipping (may be bundled in flashinfer-python)."
fi

echo "Installing vllm-omni in dev mode..."
cd "$VLLM_OMNI_DIR"
uv pip install setuptools-scm
uv pip install -e . --no-build-isolation

echo "Done."
echo "  vllm:       $VLLM_VERSION"
echo "  flashinfer: $FLASHINFER_VERSION"
echo "  CUDA:       $CUDA_VERSION"
