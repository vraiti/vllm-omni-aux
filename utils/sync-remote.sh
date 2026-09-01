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

# `git archive` only emits an empty directory for a submodule path (it's a
# gitlink, not real content) -- without this, a repo with an initialized
# submodule (e.g. python-tracer's cpython) would silently sync an empty
# directory instead of the submodule's actual files. Recurses to handle
# submodules-of-submodules too.
archive_repo_tree() {
    local repo_dir="$1" commit="$2" dest_dir="$3"
    git -C "$repo_dir" archive "$commit" | tar -x -C "$dest_dir"

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        # `git submodule status` lines: "[-+ ]<sha> <path> [(<describe>)]"
        [[ "${line:0:1}" == "-" ]] && continue  # not initialized, nothing to sync
        local sub_path sub_commit
        sub_path="$(awk '{print $2}' <<< "$line")"
        sub_commit="$(awk '{print $1}' <<< "$line" | tr -d '+-')"
        if [[ -n "$(git -C "$repo_dir/$sub_path" status --porcelain)" ]]; then
            echo "ERROR: $repo_dir/$sub_path has uncommitted changes" >&2
            exit 1
        fi
        archive_repo_tree "$repo_dir/$sub_path" "$sub_commit" "$dest_dir/$sub_path"
    done < <(git -C "$repo_dir" submodule status 2>/dev/null)
}

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

    # A marker file (excluded from rsync's own transfer/delete) records what
    # was last synced to this remote path. Nothing but this script modifies
    # remote source trees, so a local tree whose identity matches the marker
    # is guaranteed identical to what's already there -- skip the rsync scan
    # entirely. For a git repo that identity is HEAD's commit (and the tree
    # must be clean, so HEAD fully determines the content); for a plain
    # directory there's no commit to key off of, so a hash of every file's
    # path/mtime/size stands in for it instead.
    marker_path="$REMOTE_ROOT/$repo_name/.rrr-synced-commit"
    if [[ -e "$repo_dir/.git" ]]; then
        if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
            echo "ERROR: $repo_dir has uncommitted changes" >&2
            exit 1
        fi
        local_id="$(git -C "$repo_dir" rev-parse HEAD)"
    else
        local_id="$(find "$repo_dir" -type f -printf '%P %T@ %s\n' 2>/dev/null | sort | sha256sum | awk '{print $1}')"
    fi
    remote_id="$(ssh "$SSH_ALIAS" "cat $(printf '%q' "$marker_path") 2>/dev/null" || true)"
    if [[ -n "$remote_id" && "$remote_id" == "$local_id" ]]; then
        echo "Skipping $repo_name (unchanged since last sync)"
        continue
    fi

    echo "Syncing $repo_name..."
    if [[ -e "$repo_dir/.git" ]]; then
        # Export exactly the committed tree at HEAD (via `git archive`) and
        # rsync --delete *that*, rather than the working directory --
        # gitignore-filtering the working directory still lets any
        # untracked-but-not-ignored file through, which is how a stray
        # `npm install` (e.g. agent-starter-react's 1GB+ node_modules,
        # already gitignored, but this closes the gap for anything that
        # ISN'T) could still end up on the remote. The archive only ever
        # contains what's actually committed.
        archive_dir="$(mktemp -d)"
        archive_repo_tree "$repo_dir" "$local_id" "$archive_dir"
        rsync -az --delete --exclude=.rrr-synced-commit \
            "$archive_dir/" "$SSH_ALIAS:$REMOTE_ROOT/$repo_name/"
        rm -rf "$archive_dir"
    else
        # No commit to export from -- fall back to syncing the working
        # directory directly.
        rsync -az --delete --exclude=.rrr-synced-commit \
            "$repo_dir/" "$SSH_ALIAS:$REMOTE_ROOT/$repo_name/"
    fi
    ssh "$SSH_ALIAS" "echo $(printf '%q' "$local_id") > $(printf '%q' "$marker_path")"
done 3< "$REPOS_FILE"
