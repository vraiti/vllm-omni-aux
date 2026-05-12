#!/usr/bin/env python3
import os
import re
import subprocess
import sys

VENV = "/root/venv311"
SITE_PACKAGES = os.path.join(VENV, "lib/python3.11/site-packages")
VENV_BIN = os.path.join(VENV, "bin")

os.environ["VIRTUAL_ENV"] = VENV
os.environ["PATH"] = VENV_BIN + ":" + os.environ.get("PATH", "")
sys.path.insert(0, SITE_PACKAGES)

INIT_TIMEOUT = 1800
STAGE_INIT_TIMEOUT = 900

arg_utils = os.path.join(
    SITE_PACKAGES, "vllm_omni", "engine", "arg_utils.py"
)
if not os.path.exists(arg_utils):
    arg_utils = "/root/vllm-omni/vllm_omni/engine/arg_utils.py"

src = open(arg_utils).read()
src = re.sub(
    r"(stage_init_timeout:\s*int\s*=\s*)\d+",
    rf"\g<1>{STAGE_INIT_TIMEOUT}",
    src,
)
src = re.sub(
    r"(init_timeout:\s*int\s*=\s*)\d+",
    rf"\g<1>{INIT_TIMEOUT}",
    src,
)
open(arg_utils, "w").write(src)

subprocess.run(
    [
        os.path.join(VENV_BIN, "python3"),
        "-m", "vllm_omni.entrypoints.cli.main",
        "serve", "Qwen/Qwen-Image-2512",
        "--omni", "--port", "8000",
    ],
    env=os.environ,
)
