#!/usr/bin/env bash
set -euo pipefail

# Creates the venv, installs vLLM, and sets up empty vllm-omni and
# vllm-omni-aux repositories. Run as ec2-user after setup-nvidia.sh
# and reboot.

sudo dnf install -y python3-pip python3-devel cuda-toolkit git
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
export PATH=$PATH:/usr/local/cuda/bin
pip3 install --user uv

sudo mkdir -p /app && sudo chown ec2-user:ec2-user /app

git clone https://github.com/vraiti/vllm-omni.git /app/vllm-omni
cd /app/vllm-omni
git remote add upstream https://github.com/vllm-project/vllm-omni.git

git clone https://github.com/vraiti/vllm-omni-aux.git /app/vllm-omni-aux

~/.local/bin/uv venv /app/venv --python 3.12
source /app/venv/bin/activate
uv pip install setuptools setuptools-scm vllm

echo "vLLM $(python3 -c 'import vllm; print(vllm.__version__)') installed."
echo "Activate with: source /app/venv/bin/activate"
