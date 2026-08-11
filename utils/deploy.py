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
VLLM_DIR = "/app/vllm"
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


def find_vllm_site_packages():
    result = subprocess.run(
        [os.path.join(VENV_DIR, "bin", "python3"), "-c",
         "import vllm, os; print(os.path.dirname(vllm.__file__))"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: could not locate vllm site-packages: {result.stderr.strip()}", file=sys.stderr)
        return None
    return result.stdout.strip()


def sync_vllm_source():
    if not os.path.isdir(VLLM_DIR):
        print(f"WARNING: {VLLM_DIR} not found, skipping vllm sync")
        return

    site_vllm = find_vllm_site_packages()
    if not site_vllm:
        return

    has_parent = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD~1"],
        capture_output=True, cwd=VLLM_DIR,
    ).returncode == 0
    if not has_parent:
        print("No vllm parent commit to diff against, skipping sync")
        return

    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD", "--", "vllm/"],
        capture_output=True, text=True, cwd=VLLM_DIR,
    )
    if result.returncode != 0:
        print(f"WARNING: git diff failed: {result.stderr.strip()}", file=sys.stderr)
        return

    changed = [f for f in result.stdout.splitlines()
               if f.strip() and f.startswith("vllm/") and f.endswith(".py")]
    if not changed:
        print("No vllm .py files changed")
        return

    import hashlib
    import shutil
    src_vllm = os.path.join(VLLM_DIR, "vllm")
    print(f"Checking {len(changed)} changed .py file(s)")
    copied = 0
    for rel_path in changed:
        src_path = os.path.join(VLLM_DIR, rel_path)
        rel = os.path.relpath(src_path, src_vllm)
        dst_path = os.path.join(site_vllm, rel)
        if not os.path.isfile(src_path) or not os.path.isfile(dst_path):
            continue
        src_hash = hashlib.sha256(open(src_path, "rb").read()).digest()
        dst_hash = hashlib.sha256(open(dst_path, "rb").read()).digest()
        if src_hash != dst_hash:
            shutil.copy2(src_path, dst_path)
            print(f"  updated: {rel}")
            copied += 1
    if copied:
        print(f"Synced {copied} file(s)")
    else:
        print("Site-packages already up to date")


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



def resolve_model(key):
    model = MODEL_MAP.get(key)
    if not model:
        valid = ", ".join(MODEL_MAP)
        print(f"ERROR: unknown model key '{key}' (valid: {valid})", file=sys.stderr)
        sys.exit(1)
    return model


def gpu_count():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return len([l for l in result.stdout.splitlines() if l.strip()])


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
    args = parser.parse_args()

    model = resolve_model(args.model_key)
    deploy_path = resolve_deploy_config(args.model_key, args.deploy_config)

    hf_token_path = os.path.expanduser("~/.secret/hf")
    if os.path.isfile(hf_token_path):
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    kill_gpu_processes()
    sync_vllm_source()

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Log file: {LOG_PATH}")
    vllm_bin = os.path.join(VENV_DIR, "bin", "vllm")
    serve_cmd = f"{vllm_bin} serve --omni {model} --deploy {deploy_path} --enforce-eager"
    cmd = ["bash", "-c", f"{serve_cmd} 2>&1 | tee {LOG_PATH}"]

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = VENV_DIR
    env["PATH"] = os.path.join(VENV_DIR, "bin") + ":" + env.get("PATH", "")

    proc = subprocess.Popen(cmd, start_new_session=True, env=env)

    while True:
        time.sleep(POLL_INTERVAL)

        if proc.poll() is not None:
            print(f"vllm process died with exit code {proc.returncode}", file=sys.stderr)
            return 1

        try:
            resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
            if resp.status == 200:
                print("Health check passed.")
                return 0
        except (urllib.error.URLError, OSError):
            pass


if __name__ == "__main__":
    sys.exit(main())
