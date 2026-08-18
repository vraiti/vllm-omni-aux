#!/usr/bin/env python3
"""
Decode a /v1/realtime session-history dump (written by
FullDuplexRealtimeConnection._dump_conversation_history on session
termination, e.g. /tmp/realtime_session_<id>.json) and print a full,
human-readable transcript of the conversation.

User turns store raw base64 PCM16 audio, not text -- those get transcribed
with Whisper. Assistant turns already store their own transcript text
directly, so those are just printed as-is.

Usage:
    python3 transcribe_session_history.py /tmp/realtime_session_sess_xxx.json
    python3 transcribe_session_history.py /tmp/realtime_session_sess_xxx.json \
        --whisper-model small --input-sample-rate 24000
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from math import gcd

import numpy as np

# Whisper's own audio loader shells out to ffmpeg, which isn't guaranteed to
# be installed on every box this runs on -- decode the PCM16 and resample to
# whisper's required 16kHz ourselves instead, so this has no ffmpeg dependency.
WHISPER_SAMPLE_RATE = 16000


def _pcm16_b64_to_array(audio_b64: str, sample_rate: int) -> np.ndarray:
    pcm_bytes = base64.b64decode(audio_b64)
    audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if sample_rate != WHISPER_SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = gcd(WHISPER_SAMPLE_RATE, sample_rate)
        audio = resample_poly(audio, WHISPER_SAMPLE_RATE // g, sample_rate // g).astype(np.float32)
    return audio


def _transcribe_user_item(item: dict, model, sample_rate: int) -> str:
    parts = []
    for part in item.get("content", []):
        if part.get("type") != "input_audio" or not part.get("audio"):
            continue
        audio = _pcm16_b64_to_array(part["audio"], sample_rate)
        result = model.transcribe(audio, temperature=0.0, condition_on_previous_text=False)
        text = (result.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _assistant_item_text(item: dict) -> str:
    for part in item.get("content", []):
        text = part.get("transcript") or part.get("text")
        if text:
            return text
    return ""


def _redact_content(content: list) -> list:
    """Replace any large string field (base64 audio) with its length --
    printing raw items for debugging (--raw) must never dump megabytes of
    base64 to stdout."""
    redacted = []
    for part in content:
        part = dict(part)
        for key, value in part.items():
            if isinstance(value, str) and len(value) > 64:
                part[key] = f"<{len(value)} chars>"
        redacted.append(part)
    return redacted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("history_file", help="Path to a realtime_session_<id>.json dump")
    parser.add_argument("--whisper-model", default="small", help="Whisper model size (default: small)")
    parser.add_argument(
        "--input-sample-rate",
        type=int,
        default=24000,
        help="Sample rate to assume for stored user audio (default: 24000, matching "
        "connection.py's SAMPLE_RATE_HZ used for this same base64 data server-side). "
        "If transcripts come out garbled, the actual client capture rate is likely "
        "different -- try 16000.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip Whisper transcription and print each item's raw id/status/content "
        "instead (base64 audio fields are truncated to their length) -- fast inspection "
        "of item structure/ordering without loading a model.",
    )
    args = parser.parse_args()

    with open(args.history_file) as f:
        data = json.load(f)

    items = data.get("items", [])
    if not items:
        print("No items in history file.")
        return 0

    print(f"Session: {data.get('session_id')}  Conversation: {data.get('conversation_id')}")
    if data.get("instructions"):
        print(f"Instructions: {data['instructions']}")
    print(f"{len(items)} items\n" + "=" * 60)

    if args.raw:
        for i, item in enumerate(items):
            print(f"[{i}] id={item.get('id')} role={item.get('role')} status={item.get('status')} "
                  f"type={item.get('type')} content={_redact_content(item.get('content', []))}")
        return 0

    needs_whisper = any(
        item.get("role") == "user" and any(p.get("type") == "input_audio" for p in item.get("content", []))
        for item in items
    )
    model = None
    if needs_whisper:
        import whisper

        print(f"Loading Whisper model '{args.whisper_model}'...", file=sys.stderr)
        model = whisper.load_model(args.whisper_model)

    for i, item in enumerate(items):
        role = item.get("role", "?")
        status = item.get("status", "?")
        if role == "user":
            text = _transcribe_user_item(item, model, args.input_sample_rate)
        elif role == "assistant":
            text = _assistant_item_text(item)
        else:
            text = _assistant_item_text(item)
        print(f"[{i}] {role} ({status}): {text or '<empty>'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
