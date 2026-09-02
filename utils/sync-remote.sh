#!/usr/bin/env bash
set -euo pipefail

# Syncs every repo named in a run-remote.sh profile's `sync` map (see
# profile.py -- an object of <relative path>: <label>, label one of
# default/site-package/push-only) to <remote-root> on <ssh-alias> via rsync.
# Meant to be invoked by run-remote.sh (which has already resolved the alias
# and any remote-side path expansion), but is self-contained enough to run
# standalone too. Falls back to <project-dir>/repos.txt (the same
# name[:label] format, one per line) when no --profile is given, for use
# without a profile.
ARGS=()
PROFILE_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE_NAME="$2"
            shift 2
            ;;
        --profile=*)
            PROFILE_NAME="${1#--profile=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

SSH_ALIAS="${1:?Usage: $0 <ssh-alias> <remote-root> [project-dir] [--profile NAME]}"
REMOTE_ROOT="${2:?Usage: $0 <ssh-alias> <remote-root> [project-dir] [--profile NAME]}"
PROJECT_DIR="${3:-}"

if [[ -n "$PROFILE_NAME" ]]; then
    PROFILE_PATH="$HOME/.local/run-remote/$PROFILE_NAME.json"
    if [[ ! -f "$PROFILE_PATH" ]]; then
        echo "ERROR: profile '$PROFILE_NAME' not found at $PROFILE_PATH" >&2
        exit 1
    fi
fi

# A profile's `local-home` (see profile.py) pins the project directory on
# this machine explicitly; an explicit [project-dir] argument still wins
# over it. Without either, use CWD.
if [[ -z "$PROJECT_DIR" && -n "$PROFILE_NAME" ]]; then
    PROJECT_DIR="$(jq -r '.["local-home"] // empty' "$PROFILE_PATH")"
fi
PROJECT_DIR="${PROJECT_DIR:-$PWD}"

HAVE_PROFILE_SYNC=0
if [[ -n "$PROFILE_NAME" ]]; then
    # A profile without a `sync` key falls back to repos.txt below -- only a
    # profile that actually defines `sync` (even as `{}`) uses it as-is.
    if [[ "$(jq 'has("sync")' "$PROFILE_PATH")" == "true" ]]; then
        HAVE_PROFILE_SYNC=1
        mapfile -t ENTRIES < <(jq -r '.sync | to_entries[] | "\(.key):\(.value)"' "$PROFILE_PATH")
    fi
fi

if [[ "$HAVE_PROFILE_SYNC" -eq 0 ]]; then
    REPOS_FILE="$PROJECT_DIR/repos.txt"
    if [[ ! -f "$REPOS_FILE" ]]; then
        echo "ERROR: $REPOS_FILE not found" >&2
        exit 1
    fi
    mapfile -t ENTRIES < "$REPOS_FILE"
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

# For a `:push-only` repo, the remote is expected to `git pull` its own
# copy rather than receive one via rsync (see CLAUDE.md's remote-file-editing
# protocol). Recurses into submodules so each gets pushed too.
push_repo_and_submodules() {
    local repo_dir="$1"
    if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
        echo "ERROR: $repo_dir has uncommitted changes" >&2
        return 1
    fi
    git -C "$repo_dir" push

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        [[ "${line:0:1}" == "-" ]] && continue  # not initialized, nothing to push
        local sub_path
        sub_path="$(awk '{print $2}' <<< "$line")"
        push_repo_and_submodules "$repo_dir/$sub_path"
    done < <(git -C "$repo_dir" submodule status 2>/dev/null)
}

background_pids=()

# Iterating a bash array (rather than reading lines from the source file via
# a while-read loop) sidesteps the classic gotcha where ssh/rsync inside the
# loop body would otherwise inherit the loop's stdin redirect and drain the
# rest of the input as if it were their own -- no redirect here, so no
# draining is possible.
for entry in "${ENTRIES[@]}"; do
    entry="${entry%%#*}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue

    repo_name="${entry%%:*}"
    label=""
    [[ "$entry" == *:* ]] && label="${entry#*:}"
    repo_dir="$PROJECT_DIR/$repo_name"

    if [[ ! -d "$repo_dir" ]]; then
        echo "WARNING: $repo_dir does not exist, skipping"
        continue
    fi

    if [[ "$label" == "push-only" ]]; then
        echo "Pushing $repo_name in background (push-only)..."
        (push_repo_and_submodules "$repo_dir" 2>&1 | sed "s/^/[$repo_name] /") &
        background_pids+=("$!")
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
done

if [[ ${#background_pids[@]} -gt 0 ]]; then
    wait "${background_pids[@]}"
fi
