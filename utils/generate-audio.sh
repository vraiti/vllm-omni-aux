#!/usr/bin/env bash
set -euo pipefail

HOST="${VLLM_HOST:-localhost}"
PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PROMPT="${1:-Hello, how are you today?}"
OUTPUT="${2:-output.wav}"

curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<EOF
{
  "model": "$MODEL",
  "modalities": ["text", "audio"],
  "stream": true,
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
)" | python3 -c "
import sys, json, base64

text_parts = []
audio_parts = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    chunk = json.loads(line[6:])
    modality = chunk.get('modality')
    content = None
    for choice in chunk.get('choices', []):
        delta = choice.get('delta', {})
        content = delta.get('content')
    if content is None:
        continue
    if modality == 'text':
        text_parts.append(content)
    elif modality == 'audio':
        audio_parts.append(content)

if text_parts:
    print(''.join(text_parts))
if audio_parts:
    audio_data = base64.b64decode(''.join(audio_parts))
    with open('$OUTPUT', 'wb') as f:
        f.write(audio_data)
    print(f'Audio saved to $OUTPUT ({len(audio_data)} bytes)', file=sys.stderr)
"
