#!/usr/bin/env python3
import subprocess
import sys

def main():
    result = subprocess.run(
        ["bash", "-c", "source ~/.bashrc; " + " ".join(sys.argv[1:])],
        capture_output=False,
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
