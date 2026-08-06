#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

from pathlib import Path

VLLM_OMNI_DIR = "/app/vllm-omni"
VLLM_OMNI_AUX_DIR = "/app/vllm-omni-aux"
DEPLOY_CONFIGS_DIR = os.path.join(VLLM_OMNI_AUX_DIR, "deploy-configs")
LOG_PATH = "/tmp/logs/vllm.log"
HEALTH_URL = "http://localhost:8000/health"
POLL_INTERVAL = 2

MODEL_MAP = {
    "qwen3-omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "flux2": "black-forest-labs/FLUX.2-dev",
}


def kill_gpu_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
    if not pids:
        print("No processes holding GPU memory.")
        return
    for pid in pids:
        print(f"Killing GPU process {pid}...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass



def tee_stream(stream, log_file, out):
    for line in stream:
        out.buffer.write(line)
        out.buffer.flush()
        log_file.write(line)
        log_file.flush()


def resolve_model(key):
    model = MODEL_MAP.get(key)
    if not model:
        valid = ", ".join(MODEL_MAP)
        print(f"ERROR: unknown model key '{key}' (valid: {valid})", file=sys.stderr)
        sys.exit(1)
    return model


def resolve_deploy_config(model_key, config_name):
    path = os.path.join(DEPLOY_CONFIGS_DIR, model_key, config_name)
    if not os.path.isfile(path):
        config_dir = os.path.join(DEPLOY_CONFIGS_DIR, model_key)
        print(f"ERROR: deploy config not found: {path}", file=sys.stderr)
        if os.path.isdir(config_dir):
            configs = [f.name for f in Path(config_dir).iterdir() if f.suffix == ".yaml"]
            print(f"Available configs for {model_key}:", file=sys.stderr)
            for c in configs:
                print(f"  {c}", file=sys.stderr)
        sys.exit(1)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("deploy_config")
    args = parser.parse_args()

    model = resolve_model(args.model_key)
    deploy_path = resolve_deploy_config(args.model_key, args.deploy_config)

    hf_token_path = os.path.expanduser("~/.secret/hf")
    if os.path.isfile(hf_token_path):
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    kill_gpu_processes()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    cmd = ["vllm", "serve", "--omni", model, "--deploy", deploy_path]
    log_file = open(LOG_PATH, "wb")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    tail = subprocess.Popen(
        ["tail", "-f", LOG_PATH],
        start_new_session=True,
    )

    while True:
        time.sleep(POLL_INTERVAL)

        if proc.poll() is not None:
            tail.terminate()
            log_file.close()
            print(f"vllm process died with exit code {proc.returncode}", file=sys.stderr)
            return 1

        try:
            resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
            if resp.status == 200:
                tail.terminate()
                print("Health check passed. Server running as PID %d." % proc.pid)
                return 0
        except (urllib.error.URLError, OSError):
            pass


if __name__ == "__main__":
    sys.exit(main())
