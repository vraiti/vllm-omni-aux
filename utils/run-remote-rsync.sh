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

# Run the job with `nohup ... &` so it survives a dropped network connection
# (nohup ignores the SIGHUP the shell would otherwise send it on disconnect;
# `disown` also drops it from the shell's job table so the shell exiting
# doesn't touch it either), redirected to a remote log file. Watching it is a
# separate, plain `ssh -tt ... tail -f` -- a reconnect just re-runs the
# watcher, it doesn't touch the already-running job.
EXIT_FILE="/tmp/.rrr_exit_$$_$(date +%s)"
REMOTE_LOG="/tmp/logs/rrr-$(date +%Y%m%d-%H%M%S).log"

REMOTE_SHELL_CMD+="$(printf '; echo $? > %q' "$EXIT_FILE")"
LAUNCHER_CMD="$(printf 'mkdir -p %q; : > %q; ( %s ) > %q 2>&1 < /dev/null' \
    "$(dirname "$REMOTE_LOG")" "$REMOTE_LOG" "$REMOTE_SHELL_CMD" "$REMOTE_LOG")"

# Send commands to the remote shell base64-encoded rather than via nested
# printf %q layers: ssh flattens all trailing arguments into one string and
# reparses it remotely, so every extra layer of wrapping (nohup, bash -c,
# ssh itself) needs its own %q pass, and getting one wrong silently breaks
# things (confirmed: an earlier `&` vs `&&` precedence bug backgrounded a
# whole `mkdir && truncate && job` chain instead of just the job). Base64's
# alphabet has no shell metacharacters, so it survives any number of
# reparses unmodified -- decode into a `bash -c` argument (not piped to
# bash's stdin) so `ps` still shows the real decoded command, not "bash"
# with no argv or the base64 blob itself.
launcher_b64="$(printf '%s' "$LAUNCHER_CMD" | base64 -w0)"
ssh "$SSH_ALIAS" "nohup bash -c \"\$(echo $launcher_b64 | base64 -d)\" < /dev/null > /dev/null 2>&1 & disown"

WATCH_CMD_PLAIN="$(printf 'tail -n +1 -f %q & TPID=$!; while [ ! -f %q ]; do sleep 0.5; done; sleep 0.2; kill $TPID 2>/dev/null; wait $TPID 2>/dev/null' \
    "$REMOTE_LOG" "$EXIT_FILE")"
watch_b64="$(printf '%s' "$WATCH_CMD_PLAIN" | base64 -w0)"

while true; do
    ssh -tt "$SSH_ALIAS" "exec bash -c \"\$(echo $watch_b64 | base64 -d)\""
    if ssh "$SSH_ALIAS" "test -f $(printf '%q' "$EXIT_FILE")" 2>/dev/null; then
        break
    fi
    echo "Connection to $SSH_ALIAS dropped, reconnecting in 5s..." >&2
    sleep 5
done

EXIT_CODE="$(ssh "$SSH_ALIAS" "cat $(printf '%q' "$EXIT_FILE") 2>/dev/null")"
ssh "$SSH_ALIAS" "rm -f $(printf '%q' "$EXIT_FILE")" 2>/dev/null
exit "${EXIT_CODE:-1}"
