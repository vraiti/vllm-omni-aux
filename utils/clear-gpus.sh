#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Usage: clear-gpus.sh <gpu>[,<gpu>...]"
  echo "Example: clear-gpus.sh 0,1"
  exit 1
fi

IFS=',' read -ra GPUS <<< "$1"

SHM_FILES=()

for gpu in "${GPUS[@]}"; do
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' ')
  if [ -z "$pids" ]; then
    echo "GPU $gpu: no processes"
    continue
  fi
  for pid in $pids; do
    for fd in /proc/"$pid"/fd/*; do
      target=$(readlink "$fd" 2>/dev/null) || continue
      if [[ "$target" == /dev/shm/vllm* ]]; then
        SHM_FILES+=("$target")
      fi
    done
    echo "GPU $gpu: killing PID $pid"
    kill -9 "$pid" 2>/dev/null || true
  done
done

if [ ${#SHM_FILES[@]} -gt 0 ]; then
  seen=()
  for f in "${SHM_FILES[@]}"; do
    if [[ ! " ${seen[*]:-} " =~ " $f " ]]; then
      seen+=("$f")
      rm -f "$f" && echo "Removed $f"
    fi
  done
else
  echo "No /dev/shm files to clean"
fi
