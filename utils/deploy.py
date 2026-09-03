#!/usr/bin/env python3
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

from pathlib import Path

VLLM_OMNI_AUX_DIR = str(Path(__file__).resolve().parent.parent)
PROJECT_ROOT = str(Path(VLLM_OMNI_AUX_DIR).parent)
VLLM_OMNI_DIR = os.path.join(PROJECT_ROOT, "vllm-omni")
DEPLOY_CONFIGS_DIR = os.path.join(VLLM_OMNI_AUX_DIR, "deploy-configs")
HEALTH_URL = "http://localhost:8000/health"
POLL_INTERVAL = 2

MODEL_MAP = {
    "qwen3-omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "minicpm-o": "openbmb/MiniCPM-o-4_5",
    "flux2": "black-forest-labs/FLUX.2-dev",
}

DEFAULT_TOOL_CALL_PARSER = {
    "qwen3-omni": "hermes",
}


def kill_vllm_serve_processes():
    result = subprocess.run(
        ["pgrep", "-f", "vllm serve"],
        capture_output=True, text=True,
    )
    pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
    if not pids:
        print("No lingering vllm serve processes.")
        return
    for pid in pids:
        print(f"Killing lingering vllm serve process {pid}...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


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
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Pass --enforce-eager through to vllm serve")
    parser.add_argument("--disable-tool-calling", action="store_true",
                        help="Don't pass --enable-auto-tool-choice/--tool-call-parser through to vllm serve "
                             "(tool calling is on by default when the model has a DEFAULT_TOOL_CALL_PARSER entry)")
    parser.add_argument("--tool-call-parser", default=None,
                        help="Override the tool-call parser name (defaults per model_key, see DEFAULT_TOOL_CALL_PARSER)")
    parser.add_argument("--client", default=None,
                        help="Path to a client script (e.g. vllm-omni-aux/clients/livekit_replay_client.sh) "
                             "to run once the server is healthy; the server is sent SIGINT once it exits")
    args = parser.parse_args()

    model = resolve_model(args.model_key)
    deploy_path = resolve_deploy_config(args.model_key, args.deploy_config)

    hf_token_path = os.path.expanduser("~/.secret/hf")
    if os.path.isfile(hf_token_path):
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    kill_vllm_serve_processes()
    kill_gpu_processes()

    env = os.environ.copy()
    serve_cmd = f'vllm serve --omni {model} --deploy {deploy_path}'
    if args.enforce_eager:
        serve_cmd += ' --enforce-eager'
    if not args.disable_tool_calling:
        tool_call_parser = args.tool_call_parser or DEFAULT_TOOL_CALL_PARSER.get(args.model_key)
        if tool_call_parser:
            serve_cmd += f' --enable-auto-tool-choice --tool-call-parser {tool_call_parser}'
    cmd = shlex.split(serve_cmd)

    print("deploy.py: launching vllm...")

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
            return 1

        try:
            resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
            if resp.status == 200:
                print(f"Health check passed. Server is ready.")
                break
        except (urllib.error.URLError, OSError):
            pass

    if not args.client:
        return 0

    client_path = os.path.abspath(args.client)
    if not os.path.isfile(client_path):
        print(f"ERROR: client script not found: {client_path}", file=sys.stderr)
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait()
        return 1

    print(f"deploy.py: running client {client_path}...")
    client_result = subprocess.run([client_path], cwd=os.path.dirname(client_path))

    print("deploy.py: client finished, sending SIGINT to server...")
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    proc.wait()

    return client_result.returncode


if __name__ == "__main__":
    sys.exit(main())
