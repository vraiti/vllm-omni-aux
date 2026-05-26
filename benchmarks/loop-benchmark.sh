#!/usr/bin/env bash
set -euo pipefail

# Continuously run the Qwen3-Omni benchmark to generate steady metric traffic.
# Usage: ./loop-benchmark.sh <url> [samples] [pause_seconds]

URL="${1:-http://localhost:8000}"
SAMPLES="${2:-3}"
PAUSE="${3:-5}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK="$SCRIPT_DIR/benchmark-qwen3-omni.py"
ITER=0

while true; do
    ITER=$((ITER + 1))
    echo "--- iteration $ITER $(date +%H:%M:%S) ---"
    python3 "$BENCHMARK" \
        --url "$URL" \
        --samples "$SAMPLES" \
        --output /dev/null \
    || echo "iteration $ITER failed"
    sleep "$PAUSE"
done
