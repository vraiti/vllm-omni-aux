#!/usr/bin/env bash
set -euo pipefail

INSTANCE_TYPE="${1:?Usage: $0 <instance-type> [name-suffix]}"
NAME_SUFFIX="${2:-$INSTANCE_TYPE}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

KEY_NAME="vraiti-ed25519"
SECURITY_GROUP="sg-04371316571a0bf71"
VOLUME_SIZE=200
TAG_NAME="vraiti-$(date +%Y%m%d)-vllm_omni-${NAME_SUFFIX}"

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
echo "Polling SSH readiness..."
bash "$SCRIPT_DIR/poll-ssh.sh" "$SSH_ALIAS"

SSH_ALIAS="aws"
mkdir -p ~/.ssh/config.d
ssh-keygen -R "$PUBLIC_IP" 2>/dev/null || true
cat > ~/.ssh/config.d/aws <<EOF
Host ${SSH_ALIAS}
    HostName ${PUBLIC_IP}
    User ec2-user
    IdentityFile ~/.ssh/vraiti-ed25519.pem
    StrictHostKeyChecking accept-new
EOF

BRANCH=$(git -C "$SCRIPT_DIR/../../vllm-omni" branch --show-current)
if [[ -z "$BRANCH" ]]; then
    echo "ERROR: could not determine current branch in vllm-omni" >&2
    exit 1
fi

echo "Running setup-instance.sh on $SSH_ALIAS (branch: $BRANCH)..."
ssh "$SSH_ALIAS" bash -s -- "$BRANCH" < "$SCRIPT_DIR/setup-instance.sh"

echo ""
echo "Instance: $INSTANCE_ID"
echo "IP:       $PUBLIC_IP"
echo "SSH:      ssh $SSH_ALIAS"
