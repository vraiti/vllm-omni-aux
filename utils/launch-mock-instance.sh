#!/usr/bin/env bash
# Launch a cheap, no-GPU EC2 instance for the WS-disconnect repro's mock
# server -- deliberately NOT launch-instance.sh: that script requires the
# CUDA AMI and clones/builds the full vllm/vllm-omni stack via
# setup-instance.sh, which is unnecessary (and wasteful) for a plain
# FastAPI/uvicorn mock WS server with no model behind it.
set -euo pipefail

INSTANCE_TYPE="${1:-t3.micro}"
SSH_ALIAS="${2:-ws-repro}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

KEY_NAME="vraiti-ed25519"
SECURITY_GROUP="sg-04371316571a0bf71"   # same SG as the vllm-omni box: 22/8000/9090 open
SUBNET_ID="subnet-f707ab9c"             # same subnet/VPC as the vllm-omni box
VOLUME_SIZE=8
TAG_NAME="vraiti-$(date +%Y%m%d)-ws-repro-mock"

SSH_CONFIG="$HOME/.ssh/config.d/$SSH_ALIAS"
if [[ -f "$SSH_CONFIG" ]]; then
    echo "ERROR: alias '$SSH_ALIAS' already exists at $SSH_CONFIG" >&2
    exit 1
fi

echo "Looking up latest Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [[ "$AMI_ID" == "None" || -z "$AMI_ID" ]]; then
    echo "ERROR: could not find an Amazon Linux 2023 AMI" >&2
    exit 1
fi
echo "AMI: $AMI_ID"

echo "Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET_ID" \
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$VOLUME_SIZE,VolumeType=gp3}" \
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
cat > "$SSH_CONFIG" <<EOF
# state: running
# instance-id: $INSTANCE_ID
Host ${SSH_ALIAS}
    HostName ${PUBLIC_IP}
    User ec2-user
    IdentityFile ~/.ssh/vraiti-ed25519.pem
    StrictHostKeyChecking accept-new
EOF

bash "$SCRIPT_DIR/poll-ssh.sh" "$SSH_ALIAS"

echo "Installing python3/pip + fastapi/uvicorn on $SSH_ALIAS..."
ssh "$SSH_ALIAS" "sudo dnf install -y python3-pip >/dev/null && pip3 install --user --quiet fastapi 'uvicorn[standard]' websockets"

echo ""
echo "Instance:  $INSTANCE_ID"
echo "IP:        $PUBLIC_IP"
echo "SSH:       ssh $SSH_ALIAS"
echo "Terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
