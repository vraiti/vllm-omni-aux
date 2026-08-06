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
