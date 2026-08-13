#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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

bash "$SCRIPT_DIR/sync.sh" "$SSH_ALIAS"

PROJECT_DIR="$PWD"
while [[ "$PROJECT_DIR" != "$HOME/omni" && "$PROJECT_DIR" != "/" ]]; do
    if [[ "$(dirname "$PROJECT_DIR")" == "$HOME/omni" ]]; then
        break
    fi
    PROJECT_DIR="$(dirname "$PROJECT_DIR")"
done

AUX_DIR="$PROJECT_DIR/vllm-omni-aux"

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
