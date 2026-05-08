#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://localhost:8000}"
INTERVAL="${2:-10}"

while true; do
  if curl -sf --max-time 3 "${URL}/health" > /dev/null 2>&1; then
    echo "Server is ready at ${URL}"
    exit 0
  fi
  if ! pgrep -f 'vllm-omni serve' > /dev/null 2>&1; then
    echo "Server process died" >&2
    exit 1
  fi
  sleep "$INTERVAL"
done
