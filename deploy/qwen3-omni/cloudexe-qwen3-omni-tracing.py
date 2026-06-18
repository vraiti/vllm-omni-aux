#!/usr/bin/env python3
"""CloudEXE launcher for Qwen3-Omni-30B-A3B with OpenTelemetry tracing via Jaeger."""

import os
import subprocess
import sys
import time

VENV = os.environ.get("VIRTUAL_ENV")
if not VENV:
    print("ERROR: VIRTUAL_ENV not set. Use cloudexe --pythonpath $VIRTUAL_ENV", file=sys.stderr)
    sys.exit(1)

print(f"[Launcher] Using venv: {VENV}")
VENV_BIN = os.path.join(VENV, "bin")
DEPLOY_CONFIG = "/root/projects/tracing/vllm-omni-aux/deploy/qwen3-omni/cloudexe-1xh100.yaml"
JAEGER_BINARY = "/root/jaeger/jaeger-2.19.0-linux-amd64/jaeger"
JAEGER_CONFIG = "/root/jaeger/jaeger-config.yaml"

os.environ["VIRTUAL_ENV"] = VENV
os.environ["PATH"] = VENV_BIN + ":" + os.environ.get("PATH", "")
os.environ["HOME"] = "/root"
os.environ["PYTHONUNBUFFERED"] = "1"

# Start Jaeger in background
print("[Launcher] Starting Jaeger collector on GPU node...", flush=True)
jaeger_proc = subprocess.Popen(
    [JAEGER_BINARY, f"--config={JAEGER_CONFIG}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=os.environ,
)

# Give Jaeger time to bind ports
time.sleep(5)
print("[Launcher] Jaeger started, launching vLLM-Omni...", flush=True)

# Launch vLLM-Omni with tracing enabled
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
        "--otlp-traces-endpoint", "http://localhost:4317",
        "--async-chunk",
    ],
    env=os.environ,
    stdout=sys.stdout,
    stderr=sys.stdout,
)

# Kill Jaeger on exit
print("[Launcher] Shutting down Jaeger...", flush=True)
jaeger_proc.terminate()
jaeger_proc.wait(timeout=5)

sys.exit(result.returncode)
