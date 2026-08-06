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

VLLM_OMNI_DIR = "/app/vllm-omni"
VLLM_OMNI_AUX_DIR = "/app/vllm-omni-aux"
LOG_PATH = "/tmp/vllm.log"
HEALTH_URL = "http://localhost:8000/health"
POLL_INTERVAL = 2


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy", required=True)
    args = parser.parse_args()

    kill_gpu_processes()

    cmd = ["vllm", "serve", "--omni", args.model, "--deploy", args.deploy]
    log_file = open(LOG_PATH, "wb")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    tee_thread = threading.Thread(
        target=tee_stream,
        args=(proc.stdout, log_file, sys.stdout),
        daemon=True,
    )
    tee_thread.start()

    while True:
        time.sleep(POLL_INTERVAL)

        if proc.poll() is not None:
            tee_thread.join(timeout=5)
            log_file.close()
            print(f"vllm process died with exit code {proc.returncode}", file=sys.stderr)
            return 1

        try:
            resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
            if resp.status == 200:
                print("Health check passed.")
                proc.terminate()
                proc.wait()
                tee_thread.join(timeout=5)
                log_file.close()
                return 0
        except (urllib.error.URLError, OSError):
            pass


if __name__ == "__main__":
    sys.exit(main())
