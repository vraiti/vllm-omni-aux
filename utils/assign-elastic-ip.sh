#!/usr/bin/env bash
# Allocate and associate a persistent Elastic IP for an instance, so its
# public IP survives stop/start cycles (a plain auto-assigned public IP
# does not -- confirmed the hard way when vllm-omni's IP changed from
# 3.148.174.155 to 18.223.239.121 after a restart, breaking SSH config and
# .env.local's VLLM_BASE_URL).
set -euo pipefail

SSH_ALIAS="${1:?Usage: $0 <ssh-alias>}"
INSTANCE_ID="${2:?Usage: $0 <ssh-alias> <instance-id>}"
SSH_CONFIG="$HOME/.ssh/config.d/$SSH_ALIAS"

if [[ ! -f "$SSH_CONFIG" ]]; then
    echo "ERROR: no SSH config at $SSH_CONFIG" >&2
    exit 1
fi

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text)
echo "Instance: $INSTANCE_ID (state: $STATE)"

echo "Allocating Elastic IP..."
ALLOC_ID=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
echo "Allocation: $ALLOC_ID"

echo "Associating with $INSTANCE_ID..."
aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC_ID" --output text

NEW_IP=$(aws ec2 describe-addresses --allocation-ids "$ALLOC_ID" --query 'Addresses[0].PublicIp' --output text)
echo "New (fixed) IP: $NEW_IP"

ssh-keygen -R "$NEW_IP" 2>/dev/null || true
sed -i "s/HostName .*/HostName ${NEW_IP}/" "$SSH_CONFIG"
echo "Updated $SSH_CONFIG"

echo ""
echo "Elastic IP:      $NEW_IP"
echo "Allocation ID:   $ALLOC_ID"
echo "Release later:   aws ec2 disassociate-address --association-id \$(aws ec2 describe-addresses --allocation-ids $ALLOC_ID --query 'Addresses[0].AssociationId' --output text) && aws ec2 release-address --allocation-id $ALLOC_ID"
echo ""
echo "Remember to update any .env.local / config files that hardcode the old IP."
