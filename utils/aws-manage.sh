#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SSH_CONFIG="$HOME/.ssh/config.d/aws"
SSH_ALIAS="aws"

usage() {
    cat <<EOF
Usage: $0 <command> [args]

Commands:
  create <instance-type> [name-suffix]   Launch a new instance (same as launch-instance.sh)
  start                                  Start the stopped instance and poll for SSH
  stop                                   Stop the running instance
  delete                                 Terminate the instance and remove SSH config
EOF
    exit 1
}

get_instance_id() {
    local ip
    ip=$(grep 'HostName' "$SSH_CONFIG" | head -1 | awk '{print $2}')

    local id
    id=$(aws ec2 describe-instances \
        --filters "Name=ip-address,Values=$ip" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text 2>/dev/null)

    if [[ "$id" == "None" || -z "$id" ]]; then
        id=$(aws ec2 describe-instances \
            --filters "Name=tag:Name,Values=*vllm_omni*" \
                      "Name=instance-state-name,Values=running,stopped,pending" \
            --query 'Reservations[0].Instances[0].InstanceId' \
            --output text 2>/dev/null)
    fi

    if [[ "$id" == "None" || -z "$id" ]]; then
        echo "ERROR: could not find instance" >&2
        exit 1
    fi
    echo "$id"
}

cmd_create() {
    bash "$SCRIPT_DIR/launch-instance.sh" "$@"
}

cmd_stop() {
    local id
    id=$(get_instance_id)
    echo "Stopping $id..."
    aws ec2 stop-instances --instance-ids "$id" --output text
    echo "Instance stopping."
}

cmd_start() {
    local id
    id=$(get_instance_id)
    echo "Starting $id..."
    aws ec2 start-instances --instance-ids "$id" --output text

    echo "Waiting for running state..."
    aws ec2 wait instance-running --instance-ids "$id"

    local new_ip
    new_ip=$(aws ec2 describe-instances \
        --instance-ids "$id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    echo "Public IP: $new_ip"

    ssh-keygen -R "$new_ip" 2>/dev/null || true
    sed -i "s/HostName .*/HostName $new_ip/" "$SSH_CONFIG"

    bash "$SCRIPT_DIR/poll-ssh.sh" "$SSH_ALIAS"

    echo ""
    echo "Instance: $id"
    echo "IP:       $new_ip"
    echo "SSH:      ssh $SSH_ALIAS"
}

cmd_delete() {
    local id
    id=$(get_instance_id)

    echo "Terminating $id..."
    aws ec2 terminate-instances --instance-ids "$id" --output text

    if [[ -f "$SSH_CONFIG" ]]; then
        rm "$SSH_CONFIG"
        echo "Removed $SSH_CONFIG"
    fi

    echo "Instance $id terminated, SSH alias '$SSH_ALIAS' removed."
}

COMMAND="${1:-}"
shift || true

case "$COMMAND" in
    create) cmd_create "$@" ;;
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    delete) cmd_delete ;;
    *)      usage ;;
esac
