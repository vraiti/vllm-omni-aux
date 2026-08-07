#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

usage() {
    cat <<EOF
Usage: $0 <command> [args]

Commands:
  create <instance-type> [alias]   Launch a new instance (alias defaults to 'aws')
  start [alias]                    Start the stopped instance and poll for SSH
  stop [alias]                     Stop the running instance
  delete [alias]                   Terminate the instance and remove SSH config
  mv <old-alias> <new-alias>       Re-alias the instance locally
  ls                               List SSH config entries in ~/.ssh/config.d/
EOF
    exit 1
}

INSTANCE_ALIAS="aws"
SSH_CONFIG="$HOME/.ssh/config.d/$INSTANCE_ALIAS"
SSH_ALIAS="$INSTANCE_ALIAS"

set_alias() {
    INSTANCE_ALIAS="${1:-aws}"
    SSH_CONFIG="$HOME/.ssh/config.d/$INSTANCE_ALIAS"
    SSH_ALIAS="$INSTANCE_ALIAS"
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

set_state() {
    local config="$1" state="$2"
    if grep -q '^# state:' "$config"; then
        sed -i "s/^# state:.*$/# state: $state/" "$config"
    else
        sed -i "1i# state: $state" "$config"
    fi
}

setup_sshfs() {
    mkdir -p /tmp/logs
    local mount_point="/tmp/logs/$INSTANCE_ALIAS"
    fusermount -u "$mount_point" 2>/dev/null || true
    mkdir -p "$mount_point"
    ssh "$SSH_ALIAS" "mkdir -p /tmp/logs"
    sshfs "$SSH_ALIAS:/tmp/logs" "$mount_point" -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
    echo "SSHFS:    $SSH_ALIAS:/tmp/logs -> $mount_point"
}

cmd_create() {
    local instance_type="${1:?Usage: $0 create <instance-type> [alias]}"
    set_alias "${2:-}"

    local project_dir="$PWD"
    while [[ "$project_dir" != "$HOME/omni" && "$project_dir" != "/" ]]; do
        if [[ "$(dirname "$project_dir")" == "$HOME/omni" ]]; then
            break
        fi
        project_dir="$(dirname "$project_dir")"
    done
    pushd "$project_dir" > /dev/null
    trap 'popd > /dev/null' EXIT

    if [[ -f "$SSH_CONFIG" ]]; then
        echo "ERROR: alias '$INSTANCE_ALIAS' already exists" >&2
        exit 1
    fi
    bash "$SCRIPT_DIR/launch-instance.sh" "$instance_type" "$INSTANCE_ALIAS"
    setup_sshfs
}

cmd_mv() {
    local old_alias="${1:?Usage: $0 mv <old-alias> <new-alias>}"
    local new_alias="${2:?Usage: $0 mv <old-alias> <new-alias>}"
    local old_config="$HOME/.ssh/config.d/$old_alias"
    local new_config="$HOME/.ssh/config.d/$new_alias"

    if [[ ! -f "$old_config" ]]; then
        echo "ERROR: no config at $old_config" >&2
        exit 1
    fi
    if [[ -f "$new_config" ]]; then
        echo "ERROR: alias '$new_alias' already exists" >&2
        exit 1
    fi

    local ip
    ip=$(grep 'HostName' "$old_config" | head -1 | awk '{print $2}')
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

    if [[ "$id" != "None" && -n "$id" ]]; then
        local old_name
        old_name=$(aws ec2 describe-tags \
            --filters "Name=resource-id,Values=$id" "Name=key,Values=Name" \
            --query 'Tags[0].Value' --output text 2>/dev/null)
        local new_name="${old_name%-*}-${new_alias}"
        aws ec2 create-tags --resources "$id" \
            --tags "Key=Name,Value=$new_name"
        echo "AWS Name: '$old_name' -> '$new_name'"
    else
        echo "WARNING: could not find AWS instance to rename tag" >&2
    fi

    sed -i "s/^Host .*/Host $new_alias/" "$old_config"
    mv "$old_config" "$new_config"
    echo "Renamed alias '$old_alias' -> '$new_alias'"

    local old_mount="/tmp/$old_alias"
    local new_mount="/tmp/$new_alias"
    if mountpoint -q "$old_mount" 2>/dev/null; then
        fusermount -u "$old_mount"
        rmdir "$old_mount" 2>/dev/null || true
        mkdir -p "$new_mount"
        sshfs "$new_alias:/tmp/logs" "$new_mount" -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
        echo "SSHFS:    $old_mount -> $new_mount"
    fi
}

cmd_ls() {
    for f in ~/.ssh/config.d/*; do
        [[ -f "$f" ]] || continue
        local name state
        name=$(basename "$f")
        state=$(grep -oP '^# state: \K.*' "$f" 2>/dev/null || echo "unknown")
        printf "%-20s %s\n" "$name" "$state"
    done
}

cmd_stop() {
    set_alias "${1:-}"
    local id
    id=$(get_instance_id)
    echo "Stopping $id..."
    aws ec2 stop-instances --instance-ids "$id" --output text
    set_state "$SSH_CONFIG" "stopped"
    echo "Instance stopping."
}

cmd_start() {
    set_alias "${1:-}"
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

    set_state "$SSH_CONFIG" "running"
    setup_sshfs

    echo ""
    echo "Instance: $id"
    echo "IP:       $new_ip"
    echo "SSH:      ssh $SSH_ALIAS"
}

cmd_delete() {
    set_alias "${1:-}"
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
    start)  cmd_start "$@" ;;
    stop)   cmd_stop "$@" ;;
    delete) cmd_delete "$@" ;;
    mv)     cmd_mv "$@" ;;
    ls)     cmd_ls ;;
    *)      usage ;;
esac
