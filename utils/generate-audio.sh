#!/usr/bin/env bash
set -euo pipefail

HOST="${VLLM_HOST:-localhost}"
PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PROMPT="${1:-Hello, how are you today?}"
OUTPUT="${2:-output.wav}"

response=$(curl -sf "http://${HOST}:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<EOF
{
  "model": "$MODEL",
  "modalities": ["text", "audio"],
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": "$PROMPT"}]
    }
  ]
}
EOF
)")

text=$(echo "$response" | python3 -c "
import sys, json
r = json.load(sys.stdin)
c = r['choices'][0]['message']
if c.get('content'):
    print(c['content'])
if c.get('audio') and c['audio'].get('data'):
    import base64
    audio = base64.b64decode(c['audio']['data'])
    outpath = '$OUTPUT'
    with open(outpath, 'wb') as f:
        f.write(audio)
    print(f'Audio saved to {outpath} ({len(audio)} bytes)', file=sys.stderr)
")

[ -n "$text" ] && echo "$text"
