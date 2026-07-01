#!/usr/bin/env bash
set -euo pipefail

# Usage: ./poll.sh <pid> <host:port> [interval_seconds]
# Polls a vLLM-Omni server for readiness.
# Exits 0 when the server responds 200, or 1 if the process dies first.

if [ $# -lt 2 ]; then
  echo "Usage: $0 <pid> <host:port> [interval_seconds]" >&2
  exit 2
fi

SERVER_PID="$1"
ADDR="$2"
INTERVAL="${3:-10}"

if [[ "$ADDR" =~ ^https?:// ]]; then
  echo "Error: expected host:port (e.g. localhost:8000), not a URL" >&2
  exit 2
fi

URL="http://${ADDR}"

# Get process start time in seconds since epoch
START_TIME=$(ps -p "$SERVER_PID" -o lstart= 2>/dev/null | xargs -I {} date -d "{}" +%s 2>/dev/null || echo "")

while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Process $SERVER_PID is dead"
    exit 1
  fi

  status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "${URL%/}/health" 2>/dev/null || echo 'NOCONN')

  if [ "$status" = "200" ]; then
    echo "Server ready at $URL"
    if [ -n "$START_TIME" ]; then
      END_TIME=$(date +%s)
      ELAPSED=$((END_TIME - START_TIME))
      MINUTES=$((ELAPSED / 60))
      SECONDS=$((ELAPSED % 60))
      echo "Startup time: ${MINUTES}m ${SECONDS}s (from process start)"
    fi
    exit 0
  fi

  if [ -n "$START_TIME" ]; then
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))
    UPTIME=$(printf '%dm%02ds' "$MINS" "$SECS")
  else
    UPTIME="?"
  fi
  echo "$(TZ='America/New_York' date +%H:%M:%S) +${UPTIME} pid=$SERVER_PID status=$status"
  sleep "$INTERVAL"
done
