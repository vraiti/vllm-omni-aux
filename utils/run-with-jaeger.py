#!/usr/bin/env python3
"""Launch a command with Jaeger for OpenTelemetry tracing.

This script starts Jaeger as a background subprocess and then launches the
specified command, forwarding all command-line arguments to it. When the
command exits, Jaeger is terminated.

Usage:
    python3 run-with-jaeger.py <command> [arguments...]

Example:
    python3 run-with-jaeger.py python3 -m vllm_omni.entrypoints.cli.main serve \\
        Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8000 \\
        --deploy-config config.yaml --otlp-traces-endpoint http://localhost:4317
"""

import os
import subprocess
import sys
import time

JAEGER_BINARY = "/root/jaeger/jaeger-2.19.0-linux-amd64/jaeger"
JAEGER_CONFIG = "/root/jaeger/jaeger-config.yaml"

def main():
    if len(sys.argv) < 2:
        print("Usage: run-with-jaeger.py <command> [arguments...]", file=sys.stderr)
        print("Example: run-with-jaeger.py python3 -m vllm_omni.entrypoints.cli.main serve ...", file=sys.stderr)
        sys.exit(2)

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
    print("[Launcher] Jaeger started, launching command...", flush=True)

    # Set OTLP endpoint for vLLM tracing
    env = os.environ.copy()
    if "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in env:
        env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:4317"
        print(f"[Launcher] Set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT={env['OTEL_EXPORTER_OTLP_TRACES_ENDPOINT']}", flush=True)

    # Launch command with all arguments passed to this script
    result = subprocess.run(
        sys.argv[1:],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )

    # Kill Jaeger on exit
    print("[Launcher] Shutting down Jaeger...", flush=True)
    jaeger_proc.terminate()
    try:
        jaeger_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        jaeger_proc.kill()
        jaeger_proc.wait()

    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
