#!/usr/bin/env python3
"""Extract audio and transcript from a replay client outbound log."""

import argparse
import base64
import json
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", help="Path to replay_outbound.jsonl")
    parser.add_argument("-o", "--output", default="replay_output.wav",
                        help="Output wav file (default: replay_output.wav)")
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()

    audio_chunks: list[bytes] = []
    transcript_parts: list[str] = []

    with open(args.logfile) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            data = record.get("data", {})
            typ = data.get("type", "")

            if typ in ("response.audio.delta", "response.output_audio.delta"):
                delta = data.get("delta", "")
                if delta:
                    audio_chunks.append(base64.b64decode(delta))

            elif typ in ("response.audio_transcript.delta",
                         "response.output_audio_transcript.delta"):
                delta = data.get("delta", "")
                if delta:
                    transcript_parts.append(delta)

    if transcript_parts:
        transcript = "".join(transcript_parts)
        print(f"Transcript: {transcript}")
    else:
        print("No transcript deltas found.")

    if not audio_chunks:
        print("No audio deltas found.", file=sys.stderr)
        return

    pcm = b"".join(audio_chunks)
    num_samples = len(pcm) // 2
    duration = num_samples / args.sample_rate

    with open(args.output, "wb") as out:
        # WAV header: 16-bit mono PCM
        out.write(b"RIFF")
        out.write(struct.pack("<I", 36 + len(pcm)))
        out.write(b"WAVE")
        out.write(b"fmt ")
        out.write(struct.pack("<IHHIIHH", 16, 1, 1, args.sample_rate,
                              args.sample_rate * 2, 2, 16))
        out.write(b"data")
        out.write(struct.pack("<I", len(pcm)))
        out.write(pcm)

    print(f"Wrote {args.output}: {num_samples} samples, {duration:.2f}s @ {args.sample_rate}Hz")


if __name__ == "__main__":
    main()
