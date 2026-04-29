#!/usr/bin/env bash
set -euo pipefail

POD=dev
NS=omni-demo-insecure
INTERVAL="${1:-10}"

while true; do
  ts=$(date +%H:%M:%S)

  alive=$(oc exec -n "$NS" "$POD" -- bash -c \
    "ps -p 16368 -o pid=,stat=,etime=,comm= 2>/dev/null || echo 'DEAD'")

  health=$(oc exec -n "$NS" "$POD" -- bash -c \
    "curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:8000/health 2>/dev/null || echo 'NOCONN'")

  gpu=$(oc exec -n "$NS" "$POD" -- bash -c \
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || echo 'N/A'")

  log_tail=$(oc exec -n "$NS" "$POD" -- bash -c \
    "tail -3 /tmp/vllm_server.log 2>/dev/null || echo 'no log'")

  printf '\n=== %s ===\nproc: %s\nhealth: %s\ngpu_mem_MiB: %s\nlog:\n%s\n' \
    "$ts" "$alive" "$health" "$gpu" "$log_tail"

  sleep "$INTERVAL"
done
