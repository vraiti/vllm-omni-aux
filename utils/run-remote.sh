#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <alias> <command> [args...]" >&2
    exit 1
fi

SSH_ALIAS="$1"
shift

PROJECT_DIR="$PWD"
while [[ "$PROJECT_DIR" != "$HOME/omni" && "$PROJECT_DIR" != "/" ]]; do
    if [[ "$(dirname "$PROJECT_DIR")" == "$HOME/omni" ]]; then
        break
    fi
    PROJECT_DIR="$(dirname "$PROJECT_DIR")"
done
pushd "$PROJECT_DIR" > /dev/null
trap 'popd > /dev/null' EXIT

AUX_DIR="$PROJECT_DIR/vllm-omni-aux"
OMNI_DIR="$PROJECT_DIR/vllm-omni"

auto_commit_and_push() {
    local repo_dir="$1"
    if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
        git -C "$repo_dir" add -A
        git -C "$repo_dir" commit -s -m "auto-commit $(date '+%H:%M:%S %m/%d/%Y')"
    fi
    git -C "$repo_dir" push
}

BRANCH=$(git -C "$OMNI_DIR" branch --show-current)
AUX_BRANCH=$(git -C "$AUX_DIR" branch --show-current)

auto_commit_and_push "$OMNI_DIR"
auto_commit_and_push "$AUX_DIR"

OMNI_REMOTE=$(git -C "$OMNI_DIR" config "branch.$BRANCH.remote")
AUX_REMOTE=$(git -C "$AUX_DIR" config "branch.$AUX_BRANCH.remote")

ssh "$SSH_ALIAS" "cd /app/vllm-omni && git fetch --all && git checkout $BRANCH && git reset --hard $OMNI_REMOTE/$BRANCH"
ssh "$SSH_ALIAS" "cd /app/vllm-omni-aux && git fetch --all && git checkout $AUX_BRANCH && git reset --hard $AUX_REMOTE/$AUX_BRANCH"

REMOTE_CMD="$1"
shift
mapfile -t MATCHES < <(cd "$AUX_DIR" && find . -name "$REMOTE_CMD" -type f)
if [[ ${#MATCHES[@]} -eq 1 ]]; then
    REMOTE_CMD="/app/vllm-omni-aux/${MATCHES[0]#./}"
elif [[ ${#MATCHES[@]} -gt 1 ]]; then
    echo "ERROR: ambiguous command '$REMOTE_CMD', matches:" >&2
    printf "  %s\n" "${MATCHES[@]}" >&2
    exit 1
fi

ssh -tt "$SSH_ALIAS" "$REMOTE_CMD" "$@"
