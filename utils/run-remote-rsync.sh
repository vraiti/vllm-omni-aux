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

# An alias doesn't have to live in ~/.ssh/config.d/ -- it may just be a plain
# `Host` entry in ~/.ssh/config (e.g. non-AWS boxes like a DGX Spark). Without
# this check, an unrecognized-but-real alias falls through to resolve_alias()'s
# "exactly one instance" convenience path and silently targets the WRONG host
# instead of erroring -- confirmed causing a hang against a stopped EC2 instance
# when the intended target ("spark") was only in ~/.ssh/config.
is_known_host() {
    local host="$1"
    [[ -f "$HOME/.ssh/config.d/$host" ]] && return 0
    [[ -f "$HOME/.ssh/config" ]] && grep -qE "^[[:space:]]*Host[[:space:]]+(\S+[[:space:]]+)*${host}([[:space:]]|$)" "$HOME/.ssh/config" && return 0
    return 1
}

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [alias[:remote_path]] <command> [args...]" >&2
    exit 1
fi

REMOTE_ROOT="/app"
ALIAS_CANDIDATE="${1%%:*}"
if is_known_host "$ALIAS_CANDIDATE"; then
    SSH_ALIAS="$ALIAS_CANDIDATE"
    if [[ "$1" == *:* ]]; then
        PATH_CANDIDATE="${1#*:}"
        [[ -n "$PATH_CANDIDATE" ]] && REMOTE_ROOT="$PATH_CANDIDATE"
    fi
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

REPOS_FILE="$PROJECT_DIR/repos.txt"

if [[ ! -f "$REPOS_FILE" ]]; then
    echo "ERROR: $REPOS_FILE not found" >&2
    exit 1
fi

# Detect a repo installed as a NON-editable package in $REMOTE_ROOT/venv and
# print its site-packages directory. A non-editable install is a real copy
# under site-packages, so a plain source-tree rsync to $REMOTE_ROOT/$repo_name
# (e.g. /app/vllm/) never reaches the code Python actually imports -- e.g.
# vllm pip-installed as a stock wheel instead of `pip install -e`. Prints
# nothing when the package isn't installed, or is installed editable
# (editable installs already point at the synced source tree, so no overlay
# is needed).
remote_non_editable_site_packages_dir() {
    local repo_name="$1"
    ssh "$SSH_ALIAS" "'$REMOTE_ROOT/venv/bin/python3'" "'$REMOTE_ROOT/vllm-omni-aux/utils/remote_non_editable_site_packages_dir.py'" "'$repo_name'" 2>/dev/null
}

while IFS= read -r entry || [[ -n "$entry" ]]; do
    entry="${entry%%#*}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue

    repo_name="${entry%%:*}"
    repo_dir="$PROJECT_DIR/$repo_name"

    if [[ ! -d "$repo_dir" ]]; then
        echo "WARNING: $repo_dir does not exist, skipping"
        continue
    fi

    if [[ -e "$repo_dir/.git" ]] && [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
        echo "ERROR: $repo_dir has uncommitted changes" >&2
        exit 1
    fi

    echo "Syncing $repo_name..."
    rsync -az --delete \
        "$repo_dir/" "$SSH_ALIAS:$REMOTE_ROOT/$repo_name/"

    site_packages_dir="$(remote_non_editable_site_packages_dir "$repo_name")"
    if [[ -n "$site_packages_dir" ]]; then
        # The installed package is a subdirectory of the repo (e.g.
        # vllm/vllm/*.py, not vllm/*.py at the repo root) -- sync from that
        # subdirectory, named after the site-packages target, not the repo root.
        pkg_basename="$(basename "$site_packages_dir")"
        pkg_source_dir="$repo_dir/$pkg_basename"
        if [[ ! -d "$pkg_source_dir" ]]; then
            echo "  WARNING: $repo_name looks non-editable in $REMOTE_ROOT/venv but $pkg_source_dir does not exist locally; skipping overlay" >&2
        else
            echo "  $repo_name is installed non-editable in $REMOTE_ROOT/venv; overlaying .py files onto $site_packages_dir/"
            rsync -az --prune-empty-dirs \
                --include='*/' --include='*.py' --exclude='*' \
                "$pkg_source_dir/" "$SSH_ALIAS:$site_packages_dir/"
        fi
    fi
done < "$REPOS_FILE"

AUX_DIR="$PROJECT_DIR/vllm-omni-aux"

REMOTE_CMD="$1"
shift
mapfile -t MATCHES < <(cd "$AUX_DIR" && find . -name "$REMOTE_CMD" -type f)
if [[ ${#MATCHES[@]} -eq 1 ]]; then
    REMOTE_CMD="$REMOTE_ROOT/vllm-omni-aux/${MATCHES[0]#./}"
elif [[ ${#MATCHES[@]} -gt 1 ]]; then
    echo "ERROR: ambiguous command '$REMOTE_CMD', matches:" >&2
    printf "  %s\n" "${MATCHES[@]}" >&2
    exit 1
fi

ssh "$SSH_ALIAS" "mkdir -p /opt/dlami/nvme/huggingface; mkdir -p /opt/dlami/nvme/uv"

HF_TOKEN=$(cat ~/.secret/hf)
ssh -tt "$SSH_ALIAS" HF_TOKEN="$HF_TOKEN" "$REMOTE_CMD" "$@"
