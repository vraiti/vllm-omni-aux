#!/usr/bin/env python3
"""Generate a WAV file from text using Piper TTS."""

import wave
from pathlib import Path

from piper.download_voices import download_voice
from piper.voice import PiperVoice

VOICE = "en_US-lessac-medium"
VOICE_DIR = Path(__file__).parent / "piper_voices"
OUTPUT_PATH = Path(__file__).parent / "piper_tts_output.wav"

TEXT = (
    "Hello, what is your name? "
    "What is the capital of France? "
    "What is its population?"
)


def main():
    VOICE_DIR.mkdir(exist_ok=True)
    model_path = VOICE_DIR / f"{VOICE}.onnx"

    if not model_path.exists():
        print(f"Downloading voice {VOICE}...")
        download_voice(VOICE, VOICE_DIR)

    voice = PiperVoice.load(str(model_path))

    with wave.open(str(OUTPUT_PATH), "wb") as wav_file:
        voice.synthesize_wav(TEXT, wav_file)

    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
