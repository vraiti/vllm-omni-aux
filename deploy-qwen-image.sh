#!/usr/bin/env bash
set -euo pipefail

VENV=/root/venv311
source "$VENV/bin/activate"

vllm-omni serve Qwen/Qwen-Image-2512 --omni --port 8000
