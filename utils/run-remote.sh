#!/usr/bin/env bash
set -euo pipefail

resolve_alias() {
    local configs=( ~/.ssh/config.d/* )
    local count=0
    local single=""
    for f in "${configs[@]}"; do
        [[ -f "$f" ]] || continue
        count=$((count + 1))
        single=$(basename "$f")
    done
    if [[ $count -eq 1 ]]; then
        echo "$single"
        return 0
    fi
    return 1
}

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [alias] <command> [args...]" >&2
    exit 1
fi

# If the first arg matches a config, it's the alias; otherwise try to default
if [[ -f "$HOME/.ssh/config.d/$1" ]]; then
    SSH_ALIAS="$1"
    shift
elif SSH_ALIAS=$(resolve_alias); then
    :
else
    echo "ERROR: multiple instances exist, specify an alias" >&2
    ls ~/.ssh/config.d/ >&2
    exit 1
fi

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

HF_TOKEN=$(cat ~/.secret/hf)
ssh -tt "$SSH_ALIAS" HF_TOKEN="$HF_TOKEN" "$REMOTE_CMD" "$@"
