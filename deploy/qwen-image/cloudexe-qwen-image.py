#!/usr/bin/env python3
"""CloudEXE launcher for Qwen-Image-2512 via vLLM-Omni."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _detect_venv import detect_venv

VENV = detect_venv()
VENV_BIN = os.path.join(VENV, "bin")

os.environ["VIRTUAL_ENV"] = VENV
os.environ["PATH"] = VENV_BIN + ":" + os.environ.get("PATH", "")
os.environ["HOME"] = "/root"
os.environ["PYTHONUNBUFFERED"] = "1"

result = subprocess.run(
    [
        os.path.join(VENV_BIN, "python3"),
        "-m", "vllm_omni.entrypoints.cli.main",
        "serve", "Qwen/Qwen-Image-2512",
        "--omni", "--port", "8000",
        "--init-timeout", "1800",
        "--stage-init-timeout", "1800",
    ],
    env=os.environ,
    stdout=sys.stdout,
    stderr=sys.stdout,
)
sys.exit(result.returncode)
