# AWS GPU Instance Deployment

## Launching

```bash
aws ec2 run-instances \
  --image-id ami-0806afd7f0392af4d \
  --instance-type g7e.2xlarge \
  --key-name vraiti-ed25519 \
  --security-group-ids sg-04371316571a0bf71 \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=vraiti-$(date +%Y%m%d)-vllm_omni-<experiment>}]" \
  --query 'Instances[0].InstanceId' \
  --output text
```

- `vraiti-rhel10-vllm-omni` is a pre-built RHEL 10 AMI (50 GiB) with
  NVIDIA drivers, Python 3.12, cuda-toolkit, uv, vllm, NVMe instance
  store mount service, and `~/.cache` symlinked to the NVMe volume.
  Empty git repos at `/app/vllm-omni` and `/app/vllm-omni-aux` with
  remotes configured. Skip to **Serving** when using this AMI.
  Look up the AMI ID with:
  ```bash
  aws ec2 describe-images --owners self --filters "Name=tag:Name,Values=vraiti-rhel10-vllm-omni" \
    --query 'Images[0].ImageId' --output text
  ```
- `ami-0806afd7f0392af4d` is RHEL 10.2 x86_64 base (us-east-2,
  2026-06-18). Requires building the AMI (see below).
- `sg-04371316571a0bf71` is the `SSH-All` security group.
- `<experiment>` is a short name using only `[a-zA-Z_]`.
- `g7e.2xlarge` (1x RTX PRO 6000, 96 GiB). Use `g7e.12xlarge`
  (2x RTX PRO 6000) when stage separation across GPUs is needed.

Get the public IP:

```bash
aws ec2 describe-instances --instance-ids <instance-id> \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

### SSH Config

```
Host aws-g7e
    HostName <public-ip>
    User ec2-user
    IdentityFile ~/.ssh/vraiti-ed25519.pem
```

### Waiting for SSH

```bash
bash instructions/utils/poll-ssh.sh <public-ip>
```

## Building the AMI

Start from the base RHEL 10 AMI and run the two setup scripts in
order. Each script is idempotent.

### Step 1: NVIDIA drivers and NVMe automount

```bash
bash instructions/aws/setup-nvidia.sh
```

Installs NVIDIA drivers, CUDA, and a systemd service that formats and
mounts the NVMe instance store at `/opt/dlami/nvme` on boot. Symlinks
`~/.cache` to the NVMe volume. Reboots when done.

After reboot, verify with `nvidia-smi`. If the driver is not loaded,
build the DKMS module manually:

```bash
KVER=$(uname -r)
MOD_VER=$(dkms status | grep nvidia | head -1 | awk -F'[, ]+' '{print $2}')
sudo dkms build nvidia/$MOD_VER -k $KVER
sudo dkms install nvidia/$MOD_VER -k $KVER
sudo reboot
```

### Step 2: venv and repositories

```bash
bash instructions/aws/setup-vllm-omni.sh
```

Installs Python dependencies, creates a venv at `/app/venv` with
vLLM, and clones empty `vllm-omni` and `vllm-omni-aux` repos under
`/app/`. After this, snapshot the instance as an AMI.

## Determining the required vLLM version

The pre-built AMI ships a specific vLLM version, but a vLLM-Omni
branch may require a different one.  To find the version the branch
expects, search for the base image tag it references:

```bash
grep -rE 'v0\.[0-9]+\.[0-9]+' /app/vllm-omni --include='*.py' --include='*.yaml' --include='*.toml' | head -20
```

Look for strings like `vllm/vllm-openai:v0.24.0` or version pins in
config files.  If the branch expects a different version than what is
installed (`python3 -c "import vllm; print(vllm.__version__)"`),
reinstall vLLM before installing vLLM-Omni:

```bash
source /app/venv/bin/activate
uv pip install vllm==<required-version>
```

## Installing packages

Always use `uv pip` instead of bare `pip` on AWS instances. The venv
at `/app/venv` was created with `uv` and bare `pip` installs to user
site-packages (`~/.local/`) instead of the venv, causing import
failures.

```bash
source /app/venv/bin/activate
uv pip install -e /app/vllm-omni
```

## Serving

Run the server in a bare SSH session (not tmux/screen) so that
log output streams directly and the process is easy to interrupt.

```bash
source /app/venv/bin/activate
vllm serve Qwen/Qwen3-Omni --omni \
  --deploy-config /app/vllm-omni-aux/deploy/qwen3-omni/aws-1xrtxpro6000.yaml
```

Deploy configs are in `vllm-omni-aux/deploy/qwen3-omni/`.

Poll the server until ready:

```bash
bash /app/vllm-omni-aux/utils/poll-server-health.sh <pid> http://localhost:8000/health
```

## Teardown

```bash
aws ec2 terminate-instances --instance-ids <instance-id>
```

The `SSH-All` security group and `vraiti-ed25519` key pair persist
across instances.
