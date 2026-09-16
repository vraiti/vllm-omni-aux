#!/usr/bin/env bash
set -euo pipefail

# Headless full-duplex audio client for a MiniCPM-o-Demo server, using
# examples/realtime/audio_probe.py from the MiniCPM-o-Demo repo (a
# WebSocket client against /v1/realtime -- no browser or microphone
# required). Assumes the server is already running on the given host, e.g.
# via deploy-minicpm-o-demo.sh.
#
# Usage:
#   ./minicpm-o-demo-client.sh <dev-host-ip> [input.wav] [extra audio_probe.py args...]
#
# Each run's transcript (pretty-printed JSON result, including the full
# event stream) is saved to /tmp/logs/minicpm-o-transcripts/<timestamp>.txt.
#
# Env vars:
#   GATEWAY_PORT      gateway port on the dev host (default: 8006)
#   MINICPM_DEMO_DIR  MiniCPM-o-Demo checkout (default: $PWD)

HOST="${1:?usage: $0 <dev-host-ip> [input.wav] [extra audio_probe.py args...]}"
shift

INPUT_WAV=""
if [[ $# -gt 0 && "$1" != -* ]]; then
    INPUT_WAV="$1"
    shift
fi
GATEWAY_PORT="${GATEWAY_PORT:-8006}"

DEMO_DIR="${MINICPM_DEMO_DIR:-$PWD}"
EXAMPLE_DIR="$DEMO_DIR/examples/realtime"
if [[ ! -f "$EXAMPLE_DIR/audio_probe.py" ]]; then
    echo "ERROR: audio_probe.py not found under $EXAMPLE_DIR (set MINICPM_DEMO_DIR)" >&2
    exit 1
fi
[[ -n "$INPUT_WAV" ]] || INPUT_WAV="$EXAMPLE_DIR/assets/test.wav"

VENV_DIR="$EXAMPLE_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Setting up examples/realtime venv..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q -r "$EXAMPLE_DIR/requirements.txt"
fi

LOG_DIR="/tmp/logs/minicpm-o-transcripts"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d-%H%M%S).txt"

echo "Connecting to https://$HOST:$GATEWAY_PORT (wss .../v1/realtime?mode=audio)..."
echo "Saving transcript to $LOG_FILE"

exec > >(tee "$LOG_FILE") 2>&1
exec "$VENV_DIR/bin/python" "$EXAMPLE_DIR/audio_probe.py" \
    --url "https://$HOST:$GATEWAY_PORT" \
    --input-wav "$INPUT_WAV" \
    --region "local-client" \
    --insecure \
    --pretty-json \
    --include-events \
    "$@"
