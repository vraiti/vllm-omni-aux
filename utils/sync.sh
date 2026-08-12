#!/usr/bin/env bash
set -exuo pipefail

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

if [[ -f "$HOME/.ssh/config.d/${1:-}" ]]; then
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

REPOS_FILE="$PROJECT_DIR/repos.txt"

if [[ ! -f "$REPOS_FILE" ]]; then
    echo "ERROR: $REPOS_FILE not found" >&2
    exit 1
fi

while IFS= read -r entry || [[ -n "$entry" ]]; do
    entry="${entry%%#*}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue

    repo_name="${entry%%:*}"
    suffix="${entry#"$repo_name"}"
    suffix="${suffix#:}"
    repo_dir="$PROJECT_DIR/$repo_name"

    if [[ ! -e "$repo_dir/.git" ]]; then
        echo "WARNING: $repo_dir is not a git repo, skipping"
        continue
    fi

    branch=$(git -C "$repo_dir" branch --show-current)
    remote=$(git -C "$repo_dir" config "branch.$branch.remote")

    if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
        echo "ERROR: $repo_dir has uncommitted changes" >&2
        exit 1
    fi

    git -C "$repo_dir" push

    if [[ "$suffix" == "site-package" ]]; then
        ssh -n "$SSH_ALIAS" "cd /app/$repo_name && git fetch $remote $branch && git checkout $branch && git reset --hard $remote/$branch"
    else
        ssh -n "$SSH_ALIAS" "cd /app/$repo_name && git fetch --all && git checkout $branch && git reset --hard $remote/$branch"
    fi
done < "$REPOS_FILE"
