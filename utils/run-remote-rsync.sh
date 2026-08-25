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
    # -n: this runs inside a `while read < "$REPOS_FILE"` loop, and ssh
    # without -n consumes the loop's remaining stdin (the rest of the repos
    # file), silently truncating the loop to just this one entry.
    ssh -n "$SSH_ALIAS" "'$REMOTE_ROOT/venv/bin/python3'" "'$REMOTE_ROOT/vllm-omni-aux/utils/remote_non_editable_site_packages_dir.py'" "'$repo_name'" 2>/dev/null
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

# ssh flattens all trailing arguments into a single string and reparses it
# remotely, so build one shell-safe command string (with printf %q) rather
# than passing activate/exec/env as separate ssh arguments -- an env-var
# prefix (VAR=val cmd1 && cmd2) only applies to cmd1, not cmd2, so HF_TOKEN
# must be `export`ed inside the string, not passed as a leading ssh arg.
REMOTE_SHELL_CMD="$(printf 'export HF_TOKEN=%q; source %q/venv/bin/activate && %q' "$HF_TOKEN" "$REMOTE_ROOT" "$REMOTE_CMD")"
for arg in "$@"; do
    REMOTE_SHELL_CMD+="$(printf ' %q' "$arg")"
done

# Run inside a tmux session so a dropped network connection doesn't kill the
# remote job. The job itself runs backgrounded on the remote, redirected to
# a remote log file; the tmux pane's foreground process is `tail -f` on that
# log (stopping automatically once the job's PID exits via --pid), not the
# job itself. That way, when the job finishes, its output stays behind in
# the remote log file and in the pane's scrollback -- it isn't lost to a
# tmux session/pane that exits and disappears the moment the job does.
# `tmux new-session -A` creates the session (and launches the job + tail)
# the first time; on a reconnect it just re-attaches to the still-running
# tail instead of restarting the job. The exit-code file is how we know the
# job actually finished (vs. the ssh connection merely dropping), since
# tmux's own exit status here reflects the *attach*, not the job.
SESSION_NAME="rrr-$$-$(date +%s)"
EXIT_FILE="/tmp/.rrr_exit_${SESSION_NAME}"
REMOTE_LOG="/tmp/logs/rrr.log"

REMOTE_SHELL_CMD+="$(printf '; echo $? > %q' "$EXIT_FILE")"
LAUNCHER_CMD="$(printf 'mkdir -p %q; : > %q; ( %s ) > %q 2>&1 < /dev/null & JOB_PID=$!; tail -n +1 -f --pid=$JOB_PID %q; wait $JOB_PID' \
    "$(dirname "$REMOTE_LOG")" "$REMOTE_LOG" "$REMOTE_SHELL_CMD" "$REMOTE_LOG" "$REMOTE_LOG")"
TMUX_CMD="$(printf 'tmux new-session -A -s %q %q' "$SESSION_NAME" "$LAUNCHER_CMD")"

while true; do
    ssh -tt "$SSH_ALIAS" "$TMUX_CMD"
    ssh_status=$?
    [[ "$ssh_status" -eq 0 ]] && break
    if ssh "$SSH_ALIAS" "test -f $(printf '%q' "$EXIT_FILE")" 2>/dev/null; then
        break
    fi
    echo "Connection to $SSH_ALIAS dropped, reconnecting in 5s..." >&2
    sleep 5
done

EXIT_CODE="$(ssh "$SSH_ALIAS" "cat $(printf '%q' "$EXIT_FILE") 2>/dev/null")"
ssh "$SSH_ALIAS" "rm -f $(printf '%q' "$EXIT_FILE")" 2>/dev/null
exit "${EXIT_CODE:-1}"
