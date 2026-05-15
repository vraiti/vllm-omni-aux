#!/usr/bin/env bash
set -euo pipefail

# Usage: ./poll.sh <pid> <url> [interval_seconds]
# Polls a vLLM-Omni server for readiness.
# Exits 0 when the server responds 200, or 1 if the process dies first.

if [ $# -lt 2 ]; then
  echo "Usage: $0 <pid> <url> [interval_seconds]" >&2
  exit 2
fi

SERVER_PID="$1"
URL="$2"
INTERVAL="${3:-10}"

while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Process $SERVER_PID is dead"
    exit 1
  fi

  status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$URL" 2>/dev/null || echo 'NOCONN')

  if [ "$status" = "200" ]; then
    echo "Server ready at $URL"
    exit 0
  fi

  echo "$(date +%H:%M:%S) pid=$SERVER_PID status=$status"
  sleep "$INTERVAL"
done
