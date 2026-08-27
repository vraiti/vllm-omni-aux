#!/usr/bin/env bash
set -euo pipefail

RAW=0
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--raw" ]]; then
        RAW=1
    else
        ARGS+=("$arg")
    fi
done

INSTANCE_TYPE="${ARGS[0]:?Usage: $0 <instance-type> [alias] [--raw]}"
SSH_ALIAS="${ARGS[1]:-aws}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

KEY_NAME="vraiti-ed25519"
SECURITY_GROUP="sg-04371316571a0bf71"
VOLUME_SIZE=200
TAG_NAME="vraiti-$(date +%Y%m%d)-vllm_omni-${SSH_ALIAS}"

echo "Looking up AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners self \
    --filters "Name=tag:Name,Values=vraiti-rhel10-cuda" \
    --query 'Images[0].ImageId' \
    --output text)

if [[ "$AMI_ID" == "None" || -z "$AMI_ID" ]]; then
    echo "ERROR: AMI 'vraiti-rhel10-cuda' not found" >&2
    exit 1
fi
echo "AMI: $AMI_ID"

echo "Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_SIZE,VolumeType=gp3}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_NAME}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance: $INSTANCE_ID"
echo "Waiting for instance to reach running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Public IP: $PUBLIC_IP"
mkdir -p ~/.ssh/config.d
ssh-keygen -R "$PUBLIC_IP" 2>/dev/null || true
cat > ~/.ssh/config.d/"$SSH_ALIAS" <<EOF
# state: running
Host ${SSH_ALIAS}
    HostName ${PUBLIC_IP}
    User ec2-user
    IdentityFile ~/.ssh/vraiti-ed25519.pem
    StrictHostKeyChecking accept-new
EOF

bash "$SCRIPT_DIR/poll-ssh.sh" "$SSH_ALIAS"

if [[ "$RAW" -eq 1 ]]; then
    echo "Skipping install-shutdown-hook.sh (--raw)"
else
    echo "Running install-shutdown-hook.sh on $SSH_ALIAS..."
    ssh "$SSH_ALIAS" bash -s < "$SCRIPT_DIR/install-shutdown-hook.sh"
    echo "Repos aren't cloned here anymore -- sync them with run-remote-rsync.sh," \
         "then run create-venv.sh on the instance to build the venv."
fi

echo ""
echo "Instance: $INSTANCE_ID"
echo "IP:       $PUBLIC_IP"
echo "SSH:      ssh $SSH_ALIAS"
