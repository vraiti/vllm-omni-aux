#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

resolve_alias() {
    # Aliases live as `Host` lines inside files under ~/.ssh/config.d/ (e.g.
    # aws-manage's consolidated config.d/awsm, which holds one block per
    # managed instance) -- not one alias per file, so this must enumerate
    # actual Host lines, not filenames.
    local hosts
    hosts=$(grep -hoE '^Host[[:space:]]+\S+' ~/.ssh/config.d/* 2>/dev/null | awk '{print $2}' | sort -u)
    local count=0
    [[ -n "$hosts" ]] && count=$(wc -l <<< "$hosts")
    if [[ "$count" -eq 1 ]]; then
        echo "$hosts"
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
    grep -qE "^Host[[:space:]]+(\S+[[:space:]]+)*${host}([[:space:]]|\$)" ~/.ssh/config.d/* 2>/dev/null && return 0
    [[ -f "$HOME/.ssh/config" ]] && grep -qE "^[[:space:]]*Host[[:space:]]+(\S+[[:space:]]+)*${host}([[:space:]]|$)" "$HOME/.ssh/config" && return 0
    return 1
}

# --venv NAME can appear anywhere in the argument list; it selects the venv
# directory to activate remotely (relative to REMOTE_ROOT), defaulting to
# "venv" for hosts provisioned the standard way. --env VAR=value (repeatable)
# exports additional environment variables for the remote command, alongside
# the always-exported HF_TOKEN.
VENV_NAME="venv"
VENV_SPECIFIED=0
EXTRA_ENV=()
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV_NAME="$2"
            VENV_SPECIFIED=1
            shift 2
            ;;
        --venv=*)
            VENV_NAME="${1#--venv=}"
            VENV_SPECIFIED=1
            shift
            ;;
        --env)
            EXTRA_ENV+=("$2")
            shift 2
            ;;
        --env=*)
            EXTRA_ENV+=("${1#--env=}")
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [alias[:remote_path]] [--venv NAME] [--env VAR=value ...] <command> [args...]" >&2
    exit 1
fi

# Single-quoted so the literal text ($HOME, unexpanded) survives until it's
# sent to the remote shell below -- it must expand against the remote
# user's home, not whatever $HOME happens to be on this machine.
REMOTE_ROOT='$HOME/vraiti'
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

# Resolve any remote-side expansion (e.g. $HOME) once, up front, so every
# other use of REMOTE_ROOT below is a plain, already-concrete path -- it
# doesn't need to know whether it's embedded in a raw ssh command string
# (which a remote shell would expand) or a printf %q-escaped one (which
# would escape the literal '$' and never expand it).
REMOTE_ROOT="$(ssh "$SSH_ALIAS" "echo $REMOTE_ROOT")"
ssh "$SSH_ALIAS" "mkdir -p $(printf '%q' "$REMOTE_ROOT")"

PROJECT_DIR="$PWD"
while [[ "$PROJECT_DIR" != "$HOME/omni" && "$PROJECT_DIR" != "/" ]]; do
    if [[ "$(dirname "$PROJECT_DIR")" == "$HOME/omni" ]]; then
        break
    fi
    PROJECT_DIR="$(dirname "$PROJECT_DIR")"
done

# Push each repo listed in repos.txt to its git remote in the background --
# started here (before the potentially-slow sync/venv/deploy work below) and
# collected right before this script exits, so a slow/stalled `git push`
# never blocks the actual remote work, but its result is still surfaced.
PUSH_LOG_DIR="$(mktemp -d)"
PUSH_PIDS=()
PUSH_REPO_NAMES=()
if [[ -f "$PROJECT_DIR/repos.txt" ]]; then
    while IFS= read -r entry <&3 || [[ -n "$entry" ]]; do
        entry="${entry%%#*}"
        entry="${entry// /}"
        [[ -z "$entry" ]] && continue
        repo_name="${entry%%:*}"
        repo_dir="$PROJECT_DIR/$repo_name"
        [[ -d "$repo_dir/.git" ]] || continue
        # Site-package repos (e.g. vllm) are intentionally pinned to a
        # detached HEAD at a specific upstream commit -- nothing to push,
        # and `git push` there is a hard error, not a real failure.
        if ! git -C "$repo_dir" symbolic-ref -q HEAD >/dev/null 2>&1; then
            continue
        fi
        git -C "$repo_dir" push > "$PUSH_LOG_DIR/$repo_name.log" 2>&1 &
        PUSH_PIDS+=("$!")
        PUSH_REPO_NAMES+=("$repo_name")
    done 3< "$PROJECT_DIR/repos.txt"
fi

report_pushes() {
    for i in "${!PUSH_PIDS[@]}"; do
        if wait "${PUSH_PIDS[$i]}"; then
            :
        else
            echo "git push (${PUSH_REPO_NAMES[$i]}) FAILED:" >&2
            cat "$PUSH_LOG_DIR/${PUSH_REPO_NAMES[$i]}.log" >&2
        fi
    done
    rm -rf "$PUSH_LOG_DIR"
}
trap report_pushes EXIT

bash "$SCRIPT_DIR/sync-remote.sh" "$SSH_ALIAS" "$REMOTE_ROOT" "$PROJECT_DIR"

REMOTE_VENV_DIR="$REMOTE_ROOT/$VENV_NAME"
if ! ssh "$SSH_ALIAS" "test -d $(printf '%q' "$REMOTE_VENV_DIR")"; then
    if [[ "$VENV_SPECIFIED" -eq 1 ]]; then
        echo "ERROR: venv '$VENV_NAME' not found at $SSH_ALIAS:$REMOTE_VENV_DIR" >&2
        exit 1
    fi
    echo "venv not found at $SSH_ALIAS:$REMOTE_VENV_DIR, creating..."
    scp "$SCRIPT_DIR/create-venv.sh" "$SSH_ALIAS:/tmp/"
    ssh "$SSH_ALIAS" "bash /tmp/create-venv.sh $(printf '%q' "$REMOTE_VENV_DIR")"
fi

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

EXTRA_EXPORTS=""
for kv in "${EXTRA_ENV[@]}"; do
    EXTRA_EXPORTS+="$(printf 'export %s=%q; ' "${kv%%=*}" "${kv#*=}")"
done

# ssh flattens all trailing arguments into a single string and reparses it
# remotely, so build one shell-safe command string (with printf %q) rather
# than passing activate/exec/env as separate ssh arguments -- an env-var
# prefix (VAR=val cmd1 && cmd2) only applies to cmd1, not cmd2, so HF_TOKEN
# (and any --env vars) must be `export`ed inside the string, not passed as a
# leading ssh arg.
REMOTE_SHELL_CMD="$(printf 'export HF_TOKEN=%q; %scd %q && source %q/bin/activate && %q' "$HF_TOKEN" "$EXTRA_EXPORTS" "$REMOTE_ROOT" "$VENV_NAME" "$REMOTE_CMD")"
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
