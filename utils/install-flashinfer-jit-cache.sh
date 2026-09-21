#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_TAG="cu${CUDA_MAJOR}0"
FLASHINFER_VERSION=$(uv pip show flashinfer-python | grep -oP '^Version: \K.*')
uv pip install "flashinfer-jit-cache==$FLASHINFER_VERSION" --index-url "https://flashinfer.ai/whl/${CUDA_TAG}"
