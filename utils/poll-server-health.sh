#!/usr/bin/env bash
set -euo pipefail

# Polls a server until its health URL returns 200, or its process dies.
#
# Usage: poll-server-health.sh <pid> <url> <log-file> [timeout-seconds] [interval-seconds]
#
# Exit 0 as soon as <url> returns HTTP 200.
# Exit 1 (printing <log-file>) if <pid> is no longer running, or if
# <timeout-seconds> elapses first.

PID="${1:?usage: $0 <pid> <url> <log-file> [timeout-seconds] [interval-seconds]}"
URL="${2:?usage: $0 <pid> <url> <log-file> [timeout-seconds] [interval-seconds]}"
LOG_FILE="${3:?usage: $0 <pid> <url> <log-file> [timeout-seconds] [interval-seconds]}"
TIMEOUT="${4:-300}"
INTERVAL="${5:-5}"

fail() {
    echo "ERROR: $1" >&2
    echo "---- $LOG_FILE ----" >&2
    cat "$LOG_FILE" >&2
    exit 1
}

elapsed=0
while (( elapsed < TIMEOUT )); do
    kill -0 "$PID" 2>/dev/null || fail "process $PID died"

    status="$(curl -sk -o /dev/null -w '%{http_code}' "$URL" || true)"
    if [[ "$status" == "200" ]]; then
        exit 0
    fi

    sleep "$INTERVAL"
    elapsed=$((elapsed + INTERVAL))
done

fail "timed out after ${TIMEOUT}s waiting for $URL"
