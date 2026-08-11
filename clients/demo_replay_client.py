#!/usr/bin/env python3
"""Replay client that converts livekit semantic-VAD captures to MiniCPM-o-Demo
duplex protocol and replays them against a Demo backend."""

import argparse
import asyncio
import base64
import json
import os
import sys
import time

import aiohttp
import numpy as np

_SESSION_DIR = os.path.dirname(__file__)
_VAD_SESSIONS = {
    "client": os.path.join(_SESSION_DIR, "livekit_session_client_vad.jsonl"),
    "semantic": os.path.join(_SESSION_DIR, "livekit_session_semantic_vad.jsonl"),
}
_REF_AUDIO_PATH = os.path.join(_SESSION_DIR, "reference-audio.wav")


def _read_wav_as_float32(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (float32 samples, sample_rate)."""
    import wave

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        width = wf.getsampwidth()
        channels = wf.getnchannels()

    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {width}")

    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]

    return samples, sr


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    ratio = dst_rate / src_rate
    n_out = int(len(samples) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(samples) - 1).astype(np.int64)
    return samples[indices]


def _pcm16_to_float32_16k(b64_audio: str, src_rate: int) -> str:
    """Convert base64 PCM16 audio at src_rate to base64 float32 PCM at 16kHz."""
    raw = base64.b64decode(b64_audio)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    resampled = _resample(samples, src_rate, 16000)
    return base64.b64encode(resampled.astype(np.float32).tobytes()).decode()


def _build_session_init(records: list, ref_audio_path: str | None) -> dict:
    """Extract system prompt from session.update messages and build session.init."""
    system_prompt = "You are a helpful voice assistant."
    for rec in records:
        sess = rec["data"].get("session", {})
        if sess.get("instructions"):
            system_prompt = sess["instructions"]
            break

    payload = {
        "system_prompt": system_prompt,
        "config": {"length_penalty": 1.1},
    }

    if ref_audio_path and os.path.isfile(ref_audio_path):
        ref_samples, ref_sr = _read_wav_as_float32(ref_audio_path)
        ref_16k = _resample(ref_samples, ref_sr, 16000)
        ref_b64 = base64.b64encode(ref_16k.astype(np.float32).tobytes()).decode()
        payload["voice"] = {
            "ref_audio_base64": ref_b64,
            "tts_ref_audio_base64": ref_b64,
        }

    return {"type": "session.init", "payload": payload}


def _convert_audio_append(msg: dict, src_rate: int) -> dict:
    """Convert input_audio_buffer.append to input.append."""
    b64_audio = msg.get("audio", "")
    converted = _pcm16_to_float32_16k(b64_audio, src_rate)
    return {
        "type": "input.append",
        "input": {
            "audio": converted,
            "force_listen": False,
        },
    }


async def _print_responses(ws, args, outbound_log):
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                typ = data.get("type", "?")
                kind = data.get("kind", "")
                detail = ""
                if kind == "audio":
                    audio = data.get("audio", "")
                    detail = f" ({len(audio)} chars b64)"
                elif kind == "text":
                    detail = f" {data.get('text', '')!r}"
                elif kind == "listen":
                    detail = " (model listening)"
                print(f"  <-  {typ} kind={kind}{detail}")
                if args.dump_responses:
                    print(f"      {json.dumps(data)}")
                record = {"ts": time.time(), "data": data}
                outbound_log.write(json.dumps(record) + "\n")
                outbound_log.flush()
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                print(f"  <-  WS closed/error: {msg}")
                break
    except asyncio.CancelledError:
        pass


async def replay(args):
    with open(args.inbound_log) as f:
        records = [json.loads(line) for line in f]

    if not records:
        print("No records in inbound log", file=sys.stderr)
        return 1

    src_rate = 24000
    for rec in records:
        sess = rec["data"].get("session", {})
        audio_in = sess.get("audio", {}).get("input", {})
        fmt = audio_in.get("format", {})
        if fmt.get("rate"):
            src_rate = int(fmt["rate"])
            break

    if args.backend:
        url = f"ws://{args.host}:{args.port}/backend"
    else:
        url = f"ws://{args.host}:{args.port}/v1/realtime?mode=audio"
    print(f"Connecting to {url}")
    print(f"Replaying {len(records)} messages (speed={args.speed}x, src_rate={src_rate})")

    session = aiohttp.ClientSession()
    try:
        ws = await session.ws_connect(url)
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        await session.close()
        return 1

    outbound_log = open(args.outbound_log, "w")
    recv_task = asyncio.create_task(_print_responses(ws, args, outbound_log))

    if not args.backend:
        # Gateway mode: wait for session.queue_done
        print("Waiting for session.queue_done...")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    typ = data.get("type", "?")
                    print(f"  <-  {typ}")
                    if typ == "session.queue_done":
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    print(f"Connection closed during init: {msg}", file=sys.stderr)
                    await session.close()
                    return 1
        except asyncio.TimeoutError:
            print("Timed out waiting for session.queue_done", file=sys.stderr)
            await ws.close()
            await session.close()
            return 1

    # Send session.init
    ref_path = args.ref_audio if args.ref_audio else _REF_AUDIO_PATH
    init_msg = _build_session_init(records, ref_path if args.voice else None)
    print(f"  -> session.init (voice={'yes' if args.voice else 'no'})")
    await ws.send_str(json.dumps(init_msg))

    # Wait for session.created
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            typ = data.get("type", "?")
            print(f"  <-  {typ}")
            if typ == "session.created":
                break

    # Replay audio chunks
    audio_records = [r for r in records if r["data"].get("type") == "input_audio_buffer.append"]
    if not audio_records:
        print("No audio records to replay", file=sys.stderr)
        recv_task.cancel()
        outbound_log.close()
        await ws.close()
        await session.close()
        return 1

    t0_capture = audio_records[0]["ts"]
    t0_wall = time.monotonic()

    for rec in audio_records:
        if args.speed > 0:
            target = t0_wall + (rec["ts"] - t0_capture) / args.speed
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        converted = _convert_audio_append(rec["data"], src_rate)
        elapsed = time.monotonic() - t0_wall
        audio_len = len(converted["input"]["audio"])
        print(f"  [{elapsed:7.3f}s] -> input.append ({audio_len} chars b64)")
        await ws.send_str(json.dumps(converted))

    if args.wait > 0:
        print(f"All messages sent. Waiting {args.wait}s for responses...")
        await asyncio.sleep(args.wait)

    # Close session
    await ws.send_str(json.dumps({"type": "session.close", "reason": "replay_done"}))
    await asyncio.sleep(1)

    recv_task.cancel()
    outbound_log.close()
    await ws.close()
    await session.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad", choices=list(_VAD_SESSIONS), default="semantic",
                        help="VAD mode session to replay (default: semantic)")
    parser.add_argument("--inbound_log", help="Path to session JSONL (overrides --vad)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (default: 22500 with --backend, 8006 otherwise)")
    parser.add_argument("--backend", action="store_true",
                        help="Connect directly to backend /backend endpoint (no gateway)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed multiplier (0 = no delay)")
    parser.add_argument("--wait", type=float, default=10.0,
                        help="Seconds to wait for responses after last message")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Total session timeout in seconds (0 = no limit)")
    parser.add_argument("--outbound-log", default="/tmp/logs/demo_replay_outbound.jsonl",
                        help="Path to write server responses")
    parser.add_argument("--dump-responses", action="store_true",
                        help="Print full JSON of server responses")
    parser.add_argument("--voice", action="store_true",
                        help="Include reference audio in session.init for voice cloning")
    parser.add_argument("--ref-audio", default=None,
                        help="Path to reference audio WAV (default: reference-audio.wav)")
    args = parser.parse_args()
    if args.port is None:
        args.port = 22500 if args.backend else 8006
    if args.inbound_log is None:
        args.inbound_log = _VAD_SESSIONS[args.vad]

    async def run():
        if args.timeout > 0:
            try:
                return await asyncio.wait_for(replay(args), timeout=args.timeout)
            except asyncio.TimeoutError:
                print(f"\nTimeout after {args.timeout}s", file=sys.stderr)
                return 1
        return await replay(args)

    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
