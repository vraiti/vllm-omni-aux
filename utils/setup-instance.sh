#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:?Usage: $0 <branch>}"

VLLM_OMNI_ORIGIN="https://github.com/vllm-project/vllm-omni.git"
VLLM_OMNI_FORK="https://github.com/vraiti/vllm-omni.git"
VLLM_OMNI_AUX_ORIGIN="https://github.com/vraiti/vllm-omni-aux.git"

echo "Cloning vllm-omni..."
git clone "$VLLM_OMNI_ORIGIN" /app/vllm-omni
cd /app/vllm-omni
git remote add fork "$VLLM_OMNI_FORK"
git fetch fork
git checkout "$BRANCH"

echo "Cloning vllm-omni-aux..."
git clone "$VLLM_OMNI_AUX_ORIGIN" /app/vllm-omni-aux

echo "Creating venv..."
uv venv /app/venv
source /app/venv/bin/activate

VLLM_VERSION=$(grep -oP 'VLLM_VERSION[= ]+v?\K[0-9]+\.[0-9]+\.[0-9]+' /app/vllm-omni/docker/Dockerfile.xpu | head -1)
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
CUDA_TAG=$(echo "$CUDA_VERSION" | tr -d '.')
echo "Installing flashinfer-jit-cache==$FLASHINFER_VERSION (cu$CUDA_TAG)..."
if ! uv pip install "flashinfer-jit-cache==$FLASHINFER_VERSION" \
    --index-url "https://flashinfer.ai/whl/cu${CUDA_TAG}" 2>/dev/null; then
    echo "flashinfer-jit-cache not available, skipping (may be bundled in flashinfer-python)."
fi

echo "Installing vllm-omni in dev mode..."
uv pip install setuptools-scm
uv pip install -e . --no-build-isolation

echo "Done."
echo "  vllm-omni branch: $BRANCH"
echo "  vllm:             $VLLM_VERSION"
echo "  flashinfer:       $FLASHINFER_VERSION"
echo "  CUDA:             $CUDA_VERSION"
