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

# Derived from this script's own location rather than hardcoded to /app, so
# this works unchanged on a project root synced elsewhere (e.g. run via
# run-remote.sh's `alias:remote_path` form onto a non-/app host).
VLLM_OMNI_AUX_DIR = str(Path(__file__).resolve().parent.parent)
PROJECT_ROOT = str(Path(VLLM_OMNI_AUX_DIR).parent)
VLLM_OMNI_DIR = os.path.join(PROJECT_ROOT, "vllm-omni")
VENV_DIR = os.path.join(PROJECT_ROOT, "venv")
DEPLOY_CONFIGS_DIR = os.path.join(VLLM_OMNI_AUX_DIR, "deploy-configs")
HEALTH_URL = "http://localhost:8000/health"
POLL_INTERVAL = 2

MODEL_MAP = {
    "qwen3-omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "minicpm-o": "openbmb/MiniCPM-o-4_5",
    "flux2": "black-forest-labs/FLUX.2-dev",
}

# Per-model default --tool-call-parser name, used unless --tool-call-parser
# is given explicitly or --disable-tool-calling is passed.
#
# qwen3-omni -> "hermes", not "qwen3_xml": confirmed directly against this
# model's own chat_template.json (HF cache), which renders tool calls as
# plain JSON inside the tags --  <tool_call>\n{"name": .., "arguments": ..}\n</tool_call>
# -- matching Hermes2ProToolParser's format exactly. qwen3_xml/Qwen3Parser
# expects a different, nested <function=name><parameter=key>value</parameter></function>
# encoding (see vllm/parser/qwen3.py's module docstring) that this model's
# template does not produce.
DEFAULT_TOOL_CALL_PARSER = {
    "qwen3-omni": "hermes",
}


def kill_vllm_serve_processes():
    # nvidia-smi's compute-apps list (kill_gpu_processes, below) only covers
    # processes holding an active CUDA context -- the top-level `vllm serve`
    # APIServer process just orchestrates the per-stage worker subprocesses
    # that actually touch the GPU, so it doesn't appear there. If those
    # workers already died (crash, previous kill_gpu_processes() run) while
    # the APIServer parent lingered, it keeps the HTTP port bound with
    # nothing left for kill_gpu_processes() to find, and the next deploy
    # fails with "Address already in use". Kill any `vllm serve` process by
    # command line directly to close that gap.
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
    env["VIRTUAL_ENV"] = VENV_DIR
    env["PATH"] = os.path.join(VENV_DIR, "bin") + ":" + env.get("PATH", "")

    vllm = os.path.join(VENV_DIR, "bin", "vllm")
    serve_cmd = f'{vllm} serve --omni {model} --deploy {deploy_path}'
    if args.enforce_eager:
        serve_cmd += ' --enforce-eager'
    if not args.disable_tool_calling:
        # On by default for any model with a DEFAULT_TOOL_CALL_PARSER entry
        # (or an explicit --tool-call-parser override) -- silently skipped
        # for models with neither, rather than failing the deploy.
        tool_call_parser = args.tool_call_parser or DEFAULT_TOOL_CALL_PARSER.get(args.model_key)
        if tool_call_parser:
            serve_cmd += f' --enable-auto-tool-choice --tool-call-parser {tool_call_parser}'
    cmd = shlex.split(serve_cmd)

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
                return 0
        except (urllib.error.URLError, OSError):
            pass


if __name__ == "__main__":
    sys.exit(main())
