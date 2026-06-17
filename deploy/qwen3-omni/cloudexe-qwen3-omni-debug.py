#!/usr/bin/env python3
"""CloudEXE launcher for Qwen3-Omni-30B-A3B via vLLM-Omni (debug/max-model-len project)."""

import os
import subprocess
import sys

# Override to use project-specific venv
VENV = "/root/projects/max-model-len/venv"
VENV_BIN = os.path.join(VENV, "bin")
DEPLOY_CONFIG = "/root/projects/max-model-len/vllm-omni-aux/deploy/qwen3-omni/cloudexe-1xh100.yaml"

os.environ["VIRTUAL_ENV"] = VENV
os.environ["PATH"] = VENV_BIN + ":" + os.environ.get("PATH", "")
os.environ["HOME"] = "/root"
os.environ["PYTHONUNBUFFERED"] = "1"

print(f"[launcher] Using venv: {VENV}")
print(f"[launcher] Using config: {DEPLOY_CONFIG}")

result = subprocess.run(
    [
        os.path.join(VENV_BIN, "python3"),
        "-m", "vllm_omni.entrypoints.cli.main",
        "serve", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "--omni", "--port", "8000",
        "--deploy-config", DEPLOY_CONFIG,
        "--log-stats",
        "--init-timeout", "1800",
        "--stage-init-timeout", "1800",
    ],
    env=os.environ,
    stdout=sys.stdout,
    stderr=sys.stdout,
)
sys.exit(result.returncode)
