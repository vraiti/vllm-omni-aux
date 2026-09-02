#!/usr/bin/env bash
set -euo pipefail

# Creates (or overwrites) a run-remote.sh profile: a JSON file under
# ~/.local/run-remote/<name>.json with a `venv` name and a table of `env`
# vars, selectable later via `run-remote.sh --profile <name>`.
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <profile-name> [--venv NAME] [--env VAR=value ...] [--host ALIAS] [--home PATH] [--include NAME ...] [--command CMD [args...]]" >&2
    exit 1
fi

PROFILE_NAME="$1"
shift

VENV_NAME=""
ENV_ARGS=()
HOST=""
HOME_PATH=""
COMMAND_ARGS=()
INCLUDE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV_NAME="$2"
            shift 2
            ;;
        --venv=*)
            VENV_NAME="${1#--venv=}"
            shift
            ;;
        --env)
            ENV_ARGS+=("$2")
            shift 2
            ;;
        --env=*)
            ENV_ARGS+=("${1#--env=}")
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --host=*)
            HOST="${1#--host=}"
            shift
            ;;
        --home)
            HOME_PATH="$2"
            shift 2
            ;;
        --home=*)
            HOME_PATH="${1#--home=}"
            shift
            ;;
        --include)
            INCLUDE_ARGS+=("$2")
            shift 2
            ;;
        --include=*)
            INCLUDE_ARGS+=("${1#--include=}")
            shift
            ;;
        --command)
            shift
            COMMAND_ARGS=("$@")
            break
            ;;
        *)
            echo "ERROR: unrecognized argument '$1'" >&2
            exit 1
            ;;
    esac
done

PROFILE_DIR="$HOME/.local/run-remote"
mkdir -p "$PROFILE_DIR"
PROFILE_PATH="$PROFILE_DIR/$PROFILE_NAME.json"

# venv/env are only written when explicitly given -- an omitted key lets a
# `--include`d profile's value show through the latest-wins merge in
# run-remote.sh instead of being clobbered by an implicit default.
env_json="{}"
for kv in "${ENV_ARGS[@]}"; do
    env_json="$(jq --arg k "${kv%%=*}" --arg v "${kv#*=}" '. + {($k): $v}' <<< "$env_json")"
done

command_json="[]"
if [[ ${#COMMAND_ARGS[@]} -gt 0 ]]; then
    command_json="$(printf '%s\n' "${COMMAND_ARGS[@]}" | jq -R . | jq -s .)"
fi

own_json="$(jq -n --arg venv "$VENV_NAME" --argjson env "$env_json" --arg host "$HOST" --arg home "$HOME_PATH" \
    --argjson command "$command_json" \
    --argjson env_specified "$([[ ${#ENV_ARGS[@]} -gt 0 ]] && echo true || echo false)" \
    '{}
     + (if $venv != "" then {venv: $venv} else {} end)
     + (if $env_specified then {env: $env} else {} end)
     + (if $host != "" then {host: $host} else {} end)
     + (if $home != "" then {home: $home} else {} end)
     + (if ($command | length) > 0 then {command: $command} else {} end)')"

# `--include` merges immediately (simple top-level `+`, latest-wins per key,
# e.g. `env` is replaced wholesale rather than deep-merged) rather than being
# stored for run-remote.sh to resolve later -- each included profile is
# already fully resolved on disk (no `include` key survives to this point),
# so this is a flat merge, applied in listed order, with this profile's own
# keys merged in last so they win over anything included.
merged_json="{}"
for inc in "${INCLUDE_ARGS[@]+"${INCLUDE_ARGS[@]}"}"; do
    inc_path="$PROFILE_DIR/$inc.json"
    if [[ ! -f "$inc_path" ]]; then
        echo "ERROR: profile '$inc' not found at $inc_path" >&2
        exit 1
    fi
    merged_json="$(jq -n --argjson a "$merged_json" --argjson b "$(cat "$inc_path")" '$a + $b')"
done
merged_json="$(jq -n --argjson a "$merged_json" --argjson b "$own_json" '$a + $b')"

echo "$merged_json" > "$PROFILE_PATH"

echo "Wrote profile '$PROFILE_NAME' to $PROFILE_PATH"
