#!/usr/bin/env bash
set -euo pipefail

AUX_DIR="$PWD/vllm-omni-aux"
OMNI_DIR="$PWD/vllm-omni"
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <alias> <command> [args...]" >&2
    exit 1
fi

SSH_ALIAS="$1"
shift

check_clean() {
    local repo_dir="$1" name="$2"
    if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
        echo "ERROR: $name has uncommitted changes" >&2
        exit 1
    fi
}

BRANCH=$(git -C "$OMNI_DIR" branch --show-current)
AUX_BRANCH=$(git -C "$AUX_DIR" branch --show-current)

check_clean "$OMNI_DIR" "vllm-omni"
check_clean "$AUX_DIR" "vllm-omni-aux"

git -C "$OMNI_DIR" push
git -C "$AUX_DIR" push

ssh "$SSH_ALIAS" "cd /app/vllm-omni && git fetch --all && git checkout $BRANCH && git pull --ff-only"
ssh "$SSH_ALIAS" "cd /app/vllm-omni-aux && git fetch --all && git checkout $AUX_BRANCH && git pull --ff-only"

ssh -tt "$SSH_ALIAS" "$@"
