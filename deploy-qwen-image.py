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
os.environ["HOME"] = "/root"
sys.path.insert(0, SITE_PACKAGES)

INIT_TIMEOUT = 1800
STAGE_INIT_TIMEOUT = 1800

OMNI_ROOT = "/root/vllm-omni/vllm_omni"

TIMEOUT_PATCHES = [
    (os.path.join(OMNI_ROOT, "engine", "arg_utils.py"), [
        (r'(stage_init_timeout:\s*int\s*=\s*)\d+', rf'\g<1>{STAGE_INIT_TIMEOUT}'),
        (r'(init_timeout:\s*int\s*=\s*)\d+', rf'\g<1>{INIT_TIMEOUT}'),
    ]),
    (os.path.join(OMNI_ROOT, "entrypoints", "omni_base.py"), [
        (r'(kwargs\.pop\("stage_init_timeout",\s*)\d+', rf'\g<1>{STAGE_INIT_TIMEOUT}'),
        (r'(kwargs\.pop\("init_timeout",\s*)\d+', rf'\g<1>{INIT_TIMEOUT}'),
    ]),
    (os.path.join(OMNI_ROOT, "engine", "async_omni_engine.py"), [
        (r'(stage_init_timeout:\s*int\s*=\s*)\d+', rf'\g<1>{STAGE_INIT_TIMEOUT}'),
        (r'(init_timeout:\s*int\s*=\s*)\d+', rf'\g<1>{INIT_TIMEOUT}'),
    ]),
]

for path, replacements in TIMEOUT_PATCHES:
    if not os.path.exists(path):
        print(f"SKIP (not found): {path}", flush=True)
        continue
    src = open(path).read()
    for pattern, repl in replacements:
        src = re.sub(pattern, repl, src)
    open(path, "w").write(src)
    print(f"Patched {path}", flush=True)

import shutil
for root, dirs, _ in os.walk(OMNI_ROOT):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
print(f"Cleared __pycache__ under {OMNI_ROOT}", flush=True)

result = subprocess.run(
    [
        os.path.join(VENV_BIN, "python3"),
        "-m", "vllm_omni.entrypoints.cli.main",
        "serve", "Qwen/Qwen-Image-2512",
        "--omni", "--port", "8000",
    ],
    env=os.environ,
    stdout=sys.stdout,
    stderr=sys.stdout,
)
sys.exit(result.returncode)
