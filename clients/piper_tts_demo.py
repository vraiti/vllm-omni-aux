#!/usr/bin/env python3
"""Generate a LiveKit replay JSONL from text sentences using Piper TTS.

Each sentence is synthesized to PCM16 24kHz audio, resampled if needed,
then chunked into input_audio_buffer.append events matching the LiveKit
capture format. Each sentence is followed by a commit + response.create
cycle, with a response.cancel before subsequent response.creates.
"""

import base64
import io
import json
import uuid
import wave
from pathlib import Path

import numpy as np
from piper.download_voices import download_voice
from piper.voice import PiperVoice

VOICE = "en_US-lessac-medium"
VOICE_DIR = Path(__file__).parent / "piper_voices"
OUTPUT_PATH = Path(__file__).parent / "replays" / "piper_tts_session.jsonl"

TARGET_SAMPLE_RATE = 24000
CHUNK_BYTES = 4800  # 100ms at 24kHz PCM16
CHUNK_INTERVAL_S = 0.1

SENTENCES = [
    "Hello, what is your name?",
    "What is the capital of France?",
    "What is its population?",
]

SESSION_UPDATES = [
    {
        "session": {
            "type": "realtime",
            "instructions": "You are a helpful voice assistant. Respond naturally and concisely.",
        },
        "type": "session.update",
        "event_id": f"instructions_update_{uuid.uuid4().hex[:12]}",
    },
    {
        "session": {
            "type": "realtime",
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "tools": [],
        },
        "type": "session.update",
        "event_id": f"tools_update_{uuid.uuid4().hex[:12]}",
    },
]


def _gen_event_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _synthesize_pcm16_24k(voice: PiperVoice, text: str) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        src_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if src_rate != TARGET_SAMPLE_RATE:
        n_out = int(len(samples) * TARGET_SAMPLE_RATE / src_rate)
        indices = np.linspace(0, len(samples) - 1, n_out)
        samples = np.interp(indices, np.arange(len(samples)), samples)
    return samples.astype(np.int16).tobytes()


def main():
    VOICE_DIR.mkdir(exist_ok=True)
    model_path = VOICE_DIR / f"{VOICE}.onnx"
    if not model_path.exists():
        print(f"Downloading voice {VOICE}...")
        download_voice(VOICE, VOICE_DIR)

    voice = PiperVoice.load(str(model_path))

    records: list[dict] = []
    ts = 1786416331.647

    for update in SESSION_UPDATES:
        records.append({"ts": ts, "data": update})
        ts += 0.001

    for i, sentence in enumerate(SENTENCES):
        pcm = _synthesize_pcm16_24k(voice, sentence)

        for offset in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[offset : offset + CHUNK_BYTES]
            if len(chunk) < CHUNK_BYTES:
                chunk += b"\x00" * (CHUNK_BYTES - len(chunk))
            records.append({
                "ts": ts,
                "data": {
                    "audio": base64.b64encode(chunk).decode(),
                    "type": "input_audio_buffer.append",
                },
            })
            ts += CHUNK_INTERVAL_S

        records.append({
            "ts": ts,
            "data": {"type": "input_audio_buffer.commit"},
        })
        ts += 0.001

        if i > 0:
            records.append({
                "ts": ts,
                "data": {"type": "response.cancel"},
            })
            ts += 0.001

        records.append({
            "ts": ts,
            "data": {
                "type": "response.create",
                "event_id": _gen_event_id("response_create"),
                "response": {
                    "instructions": None,
                    "metadata": {
                        "client_event_id": _gen_event_id("response_create"),
                    },
                },
            },
        })
        ts += 5.0

    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n_appends = sum(1 for r in records if r["data"].get("type") == "input_audio_buffer.append")
    print(f"Wrote {OUTPUT_PATH} ({len(records)} records, {n_appends} audio chunks)")


if __name__ == "__main__":
    main()
