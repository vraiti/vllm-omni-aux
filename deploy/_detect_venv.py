"""Auto-detect NVIDIA driver version and select the matching CUDA venv.

Stdlib-only (Python 3.9 compatible) so it can run under CloudEXE's
/usr/local/bin/python3 allow-list interpreter.
"""

import os
import re
import subprocess
import sys

VENV_TABLE = [
    (570, "/root/venv312", "CUDA 13.0"),
    (525, "/root/venv312-cu124", "CUDA 12.4"),
]

MIN_SUPPORTED_DRIVER = 525


def _get_driver_major():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip().splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"FATAL: Cannot detect NVIDIA driver: {exc}", file=sys.stderr)
        sys.exit(1)

    m = re.match(r"(\d+)\.", out)
    if not m:
        print(f"FATAL: Cannot parse driver version '{out}'", file=sys.stderr)
        sys.exit(1)

    major = int(m.group(1))
    print(f"[detect_venv] NVIDIA driver {out} (major={major})")
    return major


def detect_venv():
    override = os.environ.get("VLLM_VENV")
    if override:
        print(f"[detect_venv] Using VLLM_VENV override: {override}")
        return override

    major = _get_driver_major()

    if major < MIN_SUPPORTED_DRIVER:
        print(f"FATAL: Driver major {major} is too old. "
              f"Minimum supported: {MIN_SUPPORTED_DRIVER}.",
              file=sys.stderr)
        sys.exit(1)

    for min_major, venv_path, label in VENV_TABLE:
        if major >= min_major:
            print(f"[detect_venv] Selected {venv_path} ({label})")
            return venv_path

    print(f"FATAL: No venv matches driver major {major}.", file=sys.stderr)
    sys.exit(1)
