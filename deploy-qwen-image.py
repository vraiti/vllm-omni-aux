#!/usr/bin/env python3
import os
import subprocess
import sys

VENV = "/root/venv311"
SITE_PACKAGES = os.path.join(VENV, "lib/python3.11/site-packages")
VENV_BIN = os.path.join(VENV, "bin")

os.environ["VIRTUAL_ENV"] = VENV
os.environ["PATH"] = VENV_BIN + ":" + os.environ.get("PATH", "")
sys.path.insert(0, SITE_PACKAGES)

subprocess.run(
    [
        os.path.join(VENV_BIN, "python3"),
        "-m", "vllm_omni.entrypoints.cli.main",
        "serve", "Qwen/Qwen-Image-2512",
        "--omni", "--port", "8000",
    ],
    env=os.environ,
)
