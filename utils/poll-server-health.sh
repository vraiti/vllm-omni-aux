#!/usr/bin/env bash
set -euo pipefail

# Spawns <cmd> as a subprocess, redirects its output to /tmp/logs/<path>, and
# polls its PID and every port it binds at /health every 1s.
#
# Usage: poll-server-health.sh [-i] <path> -- <cmd> [args...]
#
# Exit 0 as soon as /health returns HTTP 200.
# Exit 1 (dumping /tmp/logs/<path>) if <cmd>'s process dies first.
#
# -i: tee the log to stdout instead of only redirecting to it, and keep
#     polling (without exiting) after /health returns 200 -- still exit on
#     process death.

INTERVAL=1

TEE=0
if [[ "${1:-}" == "-i" ]]; then
    TEE=1
    shift
fi

usage() {
    echo "usage: $0 [-i] <path> -- <cmd> [args...]" >&2
    exit 1
}

[[ $# -ge 1 ]] || usage
PATH_ARG="$1"
shift
[[ "${1:-}" == "--" ]] || usage
shift
[[ $# -gt 0 ]] || usage

LOG_FILE="/tmp/logs/${PATH_ARG}"
mkdir -p "$(dirname "$LOG_FILE")"

fail() {
    echo "ERROR: $1" >&2
    echo "---- $LOG_FILE ----" >&2
    cat "$LOG_FILE" >&2
    exit 1
}

if [[ "$TEE" -eq 1 ]]; then
    "$@" > >(tee "$LOG_FILE") 2>&1 &
else
    "$@" > "$LOG_FILE" 2>&1 &
fi
PID=$!

get_ports() {
    ss -H -ltnp 2>/dev/null \
        | grep -E "pid=${PID}(,|\))" \
        | awk '{print $4}' \
        | sed -E 's/.*:([0-9]+)$/\1/' \
        | sort -un || true
}

HEALTHY=0
while true; do
    kill -0 "$PID" 2>/dev/null || fail "process $PID died"

    if [[ "$HEALTHY" -eq 0 ]]; then
        for port in $(get_ports); do
            status="$(curl -sk -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/health" || true)"
            if [[ "$status" == "200" ]]; then
                HEALTHY=1
                break
            fi
        done
        if [[ "$HEALTHY" -eq 1 ]]; then
            [[ "$TEE" -eq 1 ]] || exit 0
        fi
    fi

    sleep "$INTERVAL"
done
