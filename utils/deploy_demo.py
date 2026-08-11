#!/usr/bin/env python3
"""Start the MiniCPM-o-Demo backend server on a bare-metal GPU machine.

Kills existing GPU processes, ensures config.json exists, sets up a
dedicated venv with the demo's pinned dependencies, and launches
the py_backend server with model loading and health polling.
"""

import os
import signal
import subprocess
import sys
import time

DEMO_DIR = "/app/MiniCPM-o-Demo"
VENV_DIR = os.path.join(DEMO_DIR, ".venv")
LOG_DIR = "/tmp/logs"
LOG_PATH = os.path.join(LOG_DIR, time.strftime("demo-%d%m%Y-%H%M%S.log"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "openbmb/MiniCPM-o-4_5",
)
BACKEND_PORT = 22500
POLL_INTERVAL = 5
HEALTH_URL = f"http://127.0.0.1:{BACKEND_PORT}/health"


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


def ensure_config():
    config_path = os.path.join(DEMO_DIR, "config.json")
    example_path = os.path.join(DEMO_DIR, "config.example.json")
    if not os.path.isfile(config_path):
        if os.path.isfile(example_path):
            import shutil
            shutil.copy2(example_path, config_path)
            print(f"Created config.json from {example_path}")
        else:
            print(f"WARNING: neither config.json nor config.example.json found in {DEMO_DIR}")


def ensure_venv():
    venv_python = os.path.join(VENV_DIR, "bin", "python")
    if not os.path.isfile(venv_python):
        print(f"Creating venv at {VENV_DIR}...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])

    req_file = os.path.join(DEMO_DIR, "requirements.txt")
    if os.path.isfile(req_file):
        print("Installing requirements into demo venv...")
        pip = os.path.join(VENV_DIR, "bin", "pip")
        subprocess.check_call([pip, "install", "--upgrade", "pip"])
        subprocess.check_call([
            pip, "install", "minicpmo-utils[all]>=1.0.5",
            "transformers==4.51.0", "accelerate==1.12.0",
            "torch>=2.0.0",
        ])
        subprocess.check_call([
            pip, "install", "-r", req_file,
            "--no-deps",
        ])
        subprocess.check_call([
            pip, "install",
            "fastapi>=0.128.0", "uvicorn>=0.40.0", "httpx>=0.28.0",
            "websockets>=16.0", "python-multipart",
            "pydantic>=2.11.0", "markdown>=3.6", "Pygments>=2.18.0",
            "numpy>=2.2.0", "tqdm>=4.67.0", "PyYAML>=6.0",
            "soundfile>=0.12.1",
        ])
    return venv_python


def main():
    if not os.path.isdir(DEMO_DIR):
        print(f"ERROR: {DEMO_DIR} not found", file=sys.stderr)
        sys.exit(1)

    kill_gpu_processes()
    ensure_config()
    venv_python = ensure_venv()

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Log file: {LOG_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Backend port: {BACKEND_PORT}")

    cmd = [
        venv_python, "-m", "py_backend.server",
        "--host", "0.0.0.0",
        "--port", str(BACKEND_PORT),
        "--model-path", MODEL_PATH,
    ]
    shell_cmd = " ".join(cmd) + f" 2>&1 | tee {LOG_PATH}"
    proc = subprocess.Popen(
        ["bash", "-c", shell_cmd],
        cwd=DEMO_DIR,
        start_new_session=True,
    )

    while True:
        time.sleep(POLL_INTERVAL)
        ret = proc.poll()
        if ret is not None:
            print(f"Backend exited with code {ret}")
            sys.exit(ret)
        result = subprocess.run(
            ["curl", "-sf", HEALTH_URL],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            print("Health check passed.")
            break

    sys.exit(0)


if __name__ == "__main__":
    main()
