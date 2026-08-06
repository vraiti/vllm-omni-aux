#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OMNI_DIR="$(cd "$AUX_DIR/../vllm-omni" && pwd)"
SSH_CONFIG="$HOME/.ssh/config.d/aws"

declare -A MODEL_MAP=(
    [qwen3-omni]="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    [flux2]="black-forest-labs/FLUX.2-dev"
)

MODEL_KEY="${1:?Usage: $0 <model-key> <deploy-config.yaml>}"
DEPLOY_CONFIG="${2:?Usage: $0 <model-key> <deploy-config.yaml>}"

MODEL="${MODEL_MAP[$MODEL_KEY]:-}"
if [[ -z "$MODEL" ]]; then
    echo "ERROR: unknown model key '$MODEL_KEY'" >&2
    echo "Valid keys: ${!MODEL_MAP[*]}" >&2
    exit 1
fi

DEPLOY_PATH="$AUX_DIR/deploy-configs/$MODEL_KEY/$DEPLOY_CONFIG"
if [[ ! -f "$DEPLOY_PATH" ]]; then
    echo "ERROR: deploy config not found: $DEPLOY_PATH" >&2
    echo "Available configs for $MODEL_KEY:" >&2
    find "$AUX_DIR/deploy-configs/$MODEL_KEY" -type f -name '*.yaml' -printf '  %f\n' 2>/dev/null >&2
    exit 1
fi

BRANCH=$(git -C "$OMNI_DIR" branch --show-current)
if [[ -z "$BRANCH" ]]; then
    echo "ERROR: could not determine vllm-omni branch" >&2
    exit 1
fi

AUX_BRANCH=$(git -C "$AUX_DIR" branch --show-current)
if [[ -z "$AUX_BRANCH" ]]; then
    echo "ERROR: could not determine vllm-omni-aux branch" >&2
    exit 1
fi

check_clean() {
    local repo_dir="$1" name="$2"
    if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
        echo "ERROR: $name has uncommitted changes" >&2
        exit 1
    fi
}

check_clean "$OMNI_DIR" "vllm-omni"
check_clean "$AUX_DIR" "vllm-omni-aux"

git -C "$OMNI_DIR" push
git -C "$AUX_DIR" push

HF_TOKEN=$(cat ~/.secret/hf)
SSH_ALIAS="aws"

REMOTE_DEPLOY="/app/vllm-omni-aux/deploy-configs/$MODEL_KEY/$DEPLOY_CONFIG"

echo "vllm-omni branch:     $BRANCH"
echo "vllm-omni-aux branch: $AUX_BRANCH"
echo "Deploy config:        $DEPLOY_CONFIG"
echo "Model:                $MODEL"
echo "SSH target:           $SSH_ALIAS"

ssh "$SSH_ALIAS" "cd /app/vllm-omni && git fetch --all && git checkout $BRANCH && git pull --ff-only"
ssh "$SSH_ALIAS" "cd /app/vllm-omni-aux && git fetch --all && git checkout $AUX_BRANCH && git pull --ff-only"

ssh "$SSH_ALIAS" HF_TOKEN="$HF_TOKEN" python3 /app/vllm-omni-aux/utils/deploy.py \
    --model "$MODEL" \
    --deploy "$REMOTE_DEPLOY"
