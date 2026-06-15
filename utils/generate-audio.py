#!/usr/bin/env python3
"""Send a chat completion request to vLLM-Omni and save audio output."""

import argparse
import base64
import json
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Hello, how are you today?")
    parser.add_argument("-o", "--output", default="output.wav")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--save-json", help="Save raw JSON response to file")
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = {
        "model": args.model,
        "modalities": ["text", "audio"],
        "stream": args.stream,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, "
                            "Alibaba Group, capable of perceiving auditory and visual "
                            "inputs, as well as generating text and speech."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}],
            },
        ],
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    if args.stream:
        _handle_streaming(req, args.output, args.save_json)
    else:
        _handle_non_streaming(req, args.output, args.save_json)


def _handle_streaming(req, output_path, save_json_path):
    text_parts = []
    audio_parts = []
    chunks = []

    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            chunks.append(chunk)
            modality = chunk.get("modality")
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content is None:
                    continue
                if modality == "text":
                    text_parts.append(content)
                    print(content, end="", flush=True, file=sys.stderr)
                elif modality == "audio":
                    audio_parts.append(content)

    if save_json_path:
        with open(save_json_path, "w") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"JSON saved to {save_json_path}", file=sys.stderr)

    if text_parts:
        print(file=sys.stderr)
        print("".join(text_parts))

    if audio_parts:
        audio_data = base64.b64decode("".join(audio_parts))
        with open(output_path, "wb") as f:
            f.write(audio_data)
        print(f"Audio saved to {output_path} ({len(audio_data)} bytes)", file=sys.stderr)
    else:
        print("No audio in response", file=sys.stderr)


def _handle_non_streaming(req, output_path, save_json_path):
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())

    if save_json_path:
        with open(save_json_path, "w") as f:
            json.dump(body, f, indent=2, ensure_ascii=False)
        print(f"JSON saved to {save_json_path}", file=sys.stderr)

    msg = body["choices"][0]["message"]
    print(json.dumps(msg, indent=2, ensure_ascii=False), file=sys.stderr)

    if msg.get("content"):
        print(msg["content"])

    audio = msg.get("audio")
    if audio and audio.get("data"):
        audio_data = base64.b64decode(audio["data"])
        with open(output_path, "wb") as f:
            f.write(audio_data)
        print(f"Audio saved to {output_path} ({len(audio_data)} bytes)", file=sys.stderr)
    else:
        print("No audio in response", file=sys.stderr)
        print(f"Response keys: {list(body.keys())}", file=sys.stderr)
        print(f"Message keys: {list(msg.keys())}", file=sys.stderr)
        print(f"audio field: {audio!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
