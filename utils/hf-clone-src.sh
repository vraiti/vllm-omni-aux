#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: hf-clone-src <repo_id> [dest_dir]" >&2
    echo "  Downloads a HuggingFace repo, excluding .safetensors files." >&2
    echo "  Example: hf-clone-src Qwen/Qwen3-Omni-30B-A3B-Instruct" >&2
    exit 1
fi

REPO_ID="$1"
DEST="${2:-$(basename "$REPO_ID")}"

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "https://huggingface.co/${REPO_ID}" "$DEST"

cd "$DEST"
git lfs install --local
git lfs pull --include="*" --exclude="*.safetensors,*.bin,*.pt,*.ckpt,*.gguf,*.onnx"

echo "Downloaded source tree to $DEST (large model files excluded)"
