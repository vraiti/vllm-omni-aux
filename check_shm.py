#!/usr/bin/env python3
import os
import subprocess
result = subprocess.run(["df", "-h", "/dev/shm"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
result2 = subprocess.run(["ls", "-la", "/dev/shm/"], capture_output=True, text=True)
print(result2.stdout)
print(result2.stderr)
result3 = subprocess.run(["mount"], capture_output=True, text=True)
for line in result3.stdout.splitlines():
    if "shm" in line.lower():
        print(line)
