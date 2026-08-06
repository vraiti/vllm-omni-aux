#!/usr/bin/env bash
set -euo pipefail

# Installs NVIDIA drivers, CUDA toolkit, and sets up NVMe automount.
# Run as ec2-user on a fresh RHEL 10 g7e instance. Reboots when done.

# --- NVMe instance store automount ---

sudo tee /etc/systemd/system/nvme-instance-store.service > /dev/null <<'UNIT'
[Unit]
Description=Format and mount NVMe instance store
After=local-fs.target
ConditionPathExists=!/opt/dlami/nvme/lost+found

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '\
  mkfs.ext4 -q /dev/nvme1n1 && \
  mkdir -p /opt/dlami/nvme && \
  mount /dev/nvme1n1 /opt/dlami/nvme && \
  chmod 1777 /opt/dlami/nvme'

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now nvme-instance-store.service

mkdir -p /opt/dlami/nvme/cache
ln -sfn /opt/dlami/nvme/cache ~/.cache

# --- NVIDIA drivers ---

sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-10.noarch.rpm
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/cuda-rhel10.repo
sudo dnf install -y dkms kmod-nvidia-open-dkms nvidia-driver-cuda

echo "NVIDIA driver installed. Rebooting..."
sudo reboot
