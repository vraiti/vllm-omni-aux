#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SSH_CONFIG="$HOME/.ssh/config.d/aws"

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

get_ssh_alias() {
    if [[ ! -f "$SSH_CONFIG" ]]; then
        echo "ERROR: no SSH config at $SSH_CONFIG" >&2
        exit 1
    fi
    grep '^Host ' "$SSH_CONFIG" | head -1 | awk '{print $2}'
}

get_instance_id() {
    local alias
    alias=$(get_ssh_alias)
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
        echo "ERROR: could not find instance for alias $alias" >&2
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

    local alias
    alias=$(get_ssh_alias)
    ssh-keygen -R "$new_ip" 2>/dev/null || true
    sed -i "s/HostName .*/HostName $new_ip/" "$SSH_CONFIG"

    echo "Polling SSH readiness..."
    for ((i = 1; i <= 60; i++)); do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes "$new_ip" true 2>/dev/null; then
            echo "SSH is ready."
            break
        fi
        echo "Attempt $i/60 — SSH not ready, retrying in 5s..."
        sleep 5
        if ((i == 60)); then
            echo "ERROR: SSH did not become ready after 300s" >&2
            exit 1
        fi
    done

    echo ""
    echo "Instance: $id"
    echo "IP:       $new_ip"
    echo "SSH:      ssh $alias"
}

cmd_delete() {
    local id
    id=$(get_instance_id)
    local alias
    alias=$(get_ssh_alias)

    echo "Terminating $id..."
    aws ec2 terminate-instances --instance-ids "$id" --output text

    if [[ -f "$SSH_CONFIG" ]]; then
        rm "$SSH_CONFIG"
        echo "Removed $SSH_CONFIG"
    fi

    echo "Instance $id terminated, SSH alias '$alias' removed."
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
