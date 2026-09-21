#!/usr/bin/env bash
set -euo pipefail

# Symlinks the synced aws-home-cache artifact's subdirectories into the
# locations vllm/flashinfer/Triton/NVRTC actually read and write, so cache
# output lands inside aws-home-cache and round-trips back through its
# "artifact" sync entry (see recipes/vllm-omni/base.yaml) instead of
# staying in the job's own, non-persistent checkout. Run with cwd already
# at the synced project root (true for a run-remote fini-command) -- idempotent,
# safe to run on every job, not just the first one on a given instance.
CACHE_DIR="$(pwd)/aws-home-cache"

for NAME in flashinfer nv-compute vllm; do
    mkdir -p "$HOME/.cache" "$CACHE_DIR/$NAME"
    rm -rf "$HOME/.cache/$NAME"
    ln -sfn "$CACHE_DIR/$NAME" "$HOME/.cache/$NAME"
done

for NAME in .nv .triton .humming; do
    mkdir -p "$CACHE_DIR/$NAME"
    rm -rf "$HOME/$NAME"
    ln -sfn "$CACHE_DIR/$NAME" "$HOME/$NAME"
done

echo "Linked $CACHE_DIR -> \$HOME/.cache/{flashinfer,nv-compute,vllm}, \$HOME/{.nv,.triton,.humming}"
