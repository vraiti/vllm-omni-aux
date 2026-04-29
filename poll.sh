#!/usr/bin/env bash
set -euo pipefail

# Usage: ./poll.sh [interval_seconds] [server_pid]
# Runs on the pod itself (no oc exec).

INTERVAL="${1:-10}"
SERVER_PID="${2:-}"

while true; do
  ts=$(date +%H:%M:%S)

  if [ -n "$SERVER_PID" ]; then
    alive=$(ps -p "$SERVER_PID" -o pid=,stat=,etime=,comm= 2>/dev/null || echo 'DEAD')
  else
    alive=$(pgrep -a -f 'vllm-omni|vllm_omni' 2>/dev/null | head -5 || echo 'NO MATCH')
  fi

  health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:8000/health 2>/dev/null || echo 'NOCONN')

  gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')

  log_tail=$(tail -3 /tmp/vllm_server.log 2>/dev/null || echo 'no log')

  printf '\n=== %s ===\nproc: %s\nhealth: %s\ngpu_mem_MiB: %s\nlog:\n%s\n' \
    "$ts" "$alive" "$health" "$gpu" "$log_tail"

  sleep "$INTERVAL"
done
