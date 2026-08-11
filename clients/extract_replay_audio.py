#!/usr/bin/env python3
"""Extract and concatenate audio from a proxy_inbound.jsonl replay log.

Splits audio at commit boundaries, concatenates all segments back-to-back
(no gaps), and writes a single WAV file.
"""

import argparse
import base64
import json
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inbound_log",
        nargs="?",
        default="/tmp/logs/proxy_inbound.jsonl",
        help="Path to proxy_inbound.jsonl (default: /tmp/logs/proxy_inbound.jsonl)",
    )
    parser.add_argument(
        "-o", "--output",
        default="replay_audio.wav",
        help="Output WAV file (default: replay_audio.wav)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate in Hz (default: 24000)",
    )
    args = parser.parse_args()

    with open(args.inbound_log) as f:
        msgs = [json.loads(line) for line in f]

    pcm_chunks: list[bytes] = []
    current = bytearray()
    commit_count = 0

    for m in msgs:
        d = m.get("data", {})
        t = d.get("type")
        if t == "input_audio_buffer.append":
            current.extend(base64.b64decode(d["audio"]))
        elif t == "input_audio_buffer.commit":
            if current:
                pcm_chunks.append(bytes(current))
                commit_count += 1
                current = bytearray()

    if current:
        pcm_chunks.append(bytes(current))
        commit_count += 1

    if not pcm_chunks:
        print("No audio found in replay log.", file=sys.stderr)
        return 1

    all_pcm = b"".join(pcm_chunks)
    num_samples = len(all_pcm) // 2
    duration = num_samples / args.sample_rate

    with open(args.output, "wb") as f:
        channels = 1
        sample_width = 2
        byte_rate = args.sample_rate * channels * sample_width
        block_align = channels * sample_width
        data_size = len(all_pcm)
        # WAV header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", args.sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", sample_width * 8))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(all_pcm)

    print(f"Wrote {args.output}: {commit_count} segments, {duration:.1f}s, "
          f"{args.sample_rate}Hz mono PCM16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
