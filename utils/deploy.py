#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

from pathlib import Path

VLLM_OMNI_DIR = "/app/vllm-omni"
VLLM_OMNI_AUX_DIR = "/app/vllm-omni-aux"
VENV_DIR = "/app/venv"
DEPLOY_CONFIGS_DIR = os.path.join(VLLM_OMNI_AUX_DIR, "deploy-configs")
LOG_DIR = "/tmp/logs"
LOG_PATH = os.path.join(LOG_DIR, time.strftime("vllm-%d%m%Y-%H%M%S.log"))
HEALTH_URL = "http://localhost:8000/health"
POLL_INTERVAL = 2

MODEL_MAP = {
    "qwen3-omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "minicpm-o": "openbmb/MiniCPM-o-4_5",
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


def gpu_count():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return len([l for l in result.stdout.splitlines() if l.strip()])


def resolve_model(key):
    model = MODEL_MAP.get(key)
    if not model:
        valid = ", ".join(MODEL_MAP)
        print(f"ERROR: unknown model key '{key}' (valid: {valid})", file=sys.stderr)
        sys.exit(1)
    return model


def resolve_deploy_config(model_key, config_name=None):
    if config_name is None:
        n = gpu_count()
        config_name = f"{n}gpu.yaml"
        print(f"Auto-detected {n} GPU(s), using {config_name}")
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
    parser.add_argument("deploy_config", nargs="?", default=None)
    parser.add_argument("-i", action="store_true",
                        help="Interactive: attach to server stdio, no health polling")
    args = parser.parse_args()

    model = resolve_model(args.model_key)
    deploy_path = resolve_deploy_config(args.model_key, args.deploy_config)

    hf_token_path = os.path.expanduser("~/.secret/hf")
    if os.path.isfile(hf_token_path):
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    kill_gpu_processes()

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = VENV_DIR
    env["PATH"] = os.path.join(VENV_DIR, "bin") + ":" + env.get("PATH", "")

    os.makedirs(LOG_DIR, exist_ok=True)
    Path(LOG_PATH).touch()
    stable_log = os.path.join(LOG_DIR, "vllm.log")
    try:
        os.remove(stable_log)
    except FileNotFoundError:
        pass
    os.link(LOG_PATH, stable_log)
    print(f"Log file: {LOG_PATH}")

    vllm = os.path.join(VENV_DIR, "bin", "vllm")
    serve_cmd = (
        f'{vllm} serve --omni {model} --deploy {deploy_path} --enforce-eager'
    )
    cmd = ["bash", "-c", f"{serve_cmd} 2>&1 | tee {LOG_PATH}"]

    if args.i:
        proc = subprocess.Popen(cmd, env=env)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            return 130

    proc = subprocess.Popen(cmd, start_new_session=True, env=env)

    while True:
        time.sleep(POLL_INTERVAL)

        if proc.poll() is not None:
            print(f"vllm process died with exit code {proc.returncode}", file=sys.stderr)
            print(f"Log file: {LOG_PATH}", file=sys.stderr)
            return 1

        try:
            resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
            if resp.status == 200:
                print(f"Health check passed. Server is ready.")
                print(f"Log file: {LOG_PATH}")
                return 0
        except (urllib.error.URLError, OSError):
            pass


if __name__ == "__main__":
    sys.exit(main())
