#!/usr/bin/env python3
"""CloudEXE launcher for Qwen3-Omni-30B-A3B via vLLM-Omni."""

import os
import subprocess
import sys

VENV = os.environ.get("VLLM_VENV", "/root/venv312")
VENV_BIN = os.path.join(VENV, "bin")
DEPLOY_CONFIG = "/root/vllm-omni-aux/deploy/qwen3-omni/cloudexe-1xh100.yaml"

os.environ["VIRTUAL_ENV"] = VENV
os.environ["PATH"] = VENV_BIN + ":" + os.environ.get("PATH", "")
os.environ["HOME"] = "/root"
os.environ["PYTHONUNBUFFERED"] = "1"

result = subprocess.run(
    [
        os.path.join(VENV_BIN, "python3"),
        "-m", "vllm_omni.entrypoints.cli.main",
        "serve", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "--omni", "--port", "8000",
        "--deploy-config", DEPLOY_CONFIG,
        "--init-timeout", "1800",
        "--stage-init-timeout", "1800",
    ],
    env=os.environ,
    stdout=sys.stdout,
    stderr=sys.stdout,
)
sys.exit(result.returncode)
