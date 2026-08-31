#!/usr/bin/env bash
set -euo pipefail

# Syncs every repo listed in <project-dir>/repos.txt to <remote-root> on
# <ssh-alias> via rsync. Meant to be invoked by run-remote.sh (which has
# already resolved the alias and any remote-side path expansion), but is
# self-contained enough to run standalone too.
SSH_ALIAS="${1:?Usage: $0 <ssh-alias> <remote-root> [project-dir]}"
REMOTE_ROOT="${2:?Usage: $0 <ssh-alias> <remote-root> [project-dir]}"
PROJECT_DIR="${3:-}"

if [[ -z "$PROJECT_DIR" ]]; then
    PROJECT_DIR="$PWD"
    while [[ "$PROJECT_DIR" != "$HOME/omni" && "$PROJECT_DIR" != "/" ]]; do
        if [[ "$(dirname "$PROJECT_DIR")" == "$HOME/omni" ]]; then
            break
        fi
        PROJECT_DIR="$(dirname "$PROJECT_DIR")"
    done
fi

REPOS_FILE="$PROJECT_DIR/repos.txt"

if [[ ! -f "$REPOS_FILE" ]]; then
    echo "ERROR: $REPOS_FILE not found" >&2
    exit 1
fi

# Read from fd 3, not stdin: ssh/rsync inside this loop otherwise inherit
# the loop's stdin redirect and drain the rest of repos.txt as if it were
# their own input, silently truncating the loop after the first iteration
# that calls ssh.
while IFS= read -r entry <&3 || [[ -n "$entry" ]]; do
    entry="${entry%%#*}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue

    repo_name="${entry%%:*}"
    repo_dir="$PROJECT_DIR/$repo_name"

    if [[ ! -d "$repo_dir" ]]; then
        echo "WARNING: $repo_dir does not exist, skipping"
        continue
    fi

    # A marker file (excluded from rsync's own transfer/delete) records the
    # commit last synced to this remote path. Nothing but this script
    # modifies remote source trees, and worktrees each have their own HEAD,
    # so a clean local tree whose HEAD matches the marker is guaranteed
    # identical to what's already there -- skip the rsync scan entirely.
    if [[ -e "$repo_dir/.git" ]]; then
        if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
            echo "ERROR: $repo_dir has uncommitted changes" >&2
            exit 1
        fi

        local_head="$(git -C "$repo_dir" rev-parse HEAD)"
        marker_path="$REMOTE_ROOT/$repo_name/.rrr-synced-commit"
        remote_head="$(ssh "$SSH_ALIAS" "cat $(printf '%q' "$marker_path") 2>/dev/null" || true)"
        if [[ -n "$remote_head" && "$remote_head" == "$local_head" ]]; then
            echo "Skipping $repo_name (unchanged since last sync)"
            continue
        fi
    fi

    echo "Syncing $repo_name..."
    # ':- .gitignore' merges each directory's .gitignore in as rsync excludes
    # while descending -- without this, rsync copies everything regardless
    # of gitignore, including large local-only build artifacts (e.g. a
    # 1GB+ node_modules under vllm-omni's examples/) that have no business
    # being synced to a remote instance.
    rsync -az --delete --exclude=.git --exclude=.rrr-synced-commit \
        --filter=':- .gitignore' \
        "$repo_dir/" "$SSH_ALIAS:$REMOTE_ROOT/$repo_name/"

    if [[ -e "$repo_dir/.git" ]]; then
        ssh "$SSH_ALIAS" "echo $(printf '%q' "$local_head") > $(printf '%q' "$marker_path")"
    fi
done 3< "$REPOS_FILE"
