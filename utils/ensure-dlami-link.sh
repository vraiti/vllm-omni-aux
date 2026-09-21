#!/usr/bin/env bash
set -euo pipefail

# Symlinks $HOME/.huggingface and $HOME/.uv onto the instance's ephemeral
# NVMe mount (/opt/dlami/nvme) -- large, throughput-heavy scratch (model
# downloads, package cache) that benefits from fast local storage but,
# unlike aws-home-cache, doesn't need to survive a stop/start. Idempotent --
# safe to run on every job, not just the first one on a given instance.
DLAMI_DIR="/opt/dlami/nvme"

for NAME in huggingface uv; do
    mkdir -p "$DLAMI_DIR/$NAME"
    rm -rf "$HOME/.$NAME"
    ln -sfn "$DLAMI_DIR/$NAME" "$HOME/.$NAME"
done

echo "Linked $DLAMI_DIR/{huggingface,uv} -> \$HOME/{.huggingface,.uv}"
