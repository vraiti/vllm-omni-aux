#!/usr/bin/env python3
import subprocess
import sys

sys.exit(
    subprocess.run(
        ["bash", "-c", " ".join(sys.argv[1:])],
        capture_output=False,
    ).returncode
)
