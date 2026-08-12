#!/usr/bin/env python3
"""Replay client WebSocket messages from a proxy JSONL capture."""

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import time

import aiohttp

_VOICE_NAME = "replay-ref"
_REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "reference-audio.wav")


async def _upload_voice(args):
    url = f"http://{args.host}:{args.port}/v1/audio/voices"
    form = aiohttp.FormData()
    form.add_field("name", _VOICE_NAME)
    form.add_field("consent", "replay-client")
    form.add_field(
        "audio_sample",
        open(_REF_AUDIO_PATH, "rb"),
        filename="reference-audio.wav",
        content_type="audio/wav",
    )
    async with aiohttp.ClientSession() as client:
        async with client.post(url, data=form) as resp:
            body = await resp.json()
            if resp.status == 200:
                print(f"Uploaded voice '{_VOICE_NAME}'")
            else:
                print(f"Voice upload: {resp.status} {body}")


def _build_timed_buffer(chunks, sample_rate=24000):
    """Build a mono PCM16 buffer from [(elapsed_seconds, pcm_bytes), ...]."""
    if not chunks:
        return b""
    bytes_per_sample = 2
    buf = bytearray()
    for t, data in chunks:
        offset = int(t * sample_rate) * bytes_per_sample
        end = offset + len(data)
        if end > len(buf):
            buf.extend(b"\x00" * (end - len(buf)))
        buf[offset:offset + len(data)] = data
    return bytes(buf)


def _write_stereo_wav(path, left, right, sample_rate=24000):
    """Write a stereo WAV from two mono PCM16 byte buffers."""
    left_b, right_b = bytes(left), bytes(right)
    n_left = len(left_b) // 2
    n_right = len(right_b) // 2
    n_samples = max(n_left, n_right)
    left_samples = struct.unpack(f"<{n_left}h", left_b) + (0,) * (n_samples - n_left)
    right_samples = struct.unpack(f"<{n_right}h", right_b) + (0,) * (n_samples - n_right)
    interleaved = b"".join(
        struct.pack("<hh", left_samples[i], right_samples[i])
        for i in range(n_samples)
    )
    num_channels = 2
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(interleaved)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                            byte_rate, block_align, bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(interleaved)


async def replay(args):
    with open(args.inbound_log) as f:
        records = [json.loads(line) for line in f]

    if not records:
        print("No records in inbound log", file=sys.stderr)
        return 1

    if args.voice:
        await _upload_voice(args)

    url = f"ws://{args.host}:{args.port}/v1/realtime?model={args.model}"
    print(f"Connecting to {url}")
    print(f"Replaying {len(records)} messages (speed={args.speed}x)")

    session = aiohttp.ClientSession()
    try:
        ws = await session.ws_connect(url)
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        await session.close()
        return 1

    if args.voice:
        for rec in records:
            msg = rec["data"]
            raw = json.dumps(msg)
            if "__VOICE__" in raw:
                rec["data"] = json.loads(raw.replace("__VOICE__", _VOICE_NAME))

    client_audio_chunks = []
    server_audio_chunks = []

    outbound_log = open(args.outbound_log, "w")
    t0_capture = records[0]["ts"]
    t0_wall = time.monotonic()
    recv_task = asyncio.create_task(_print_responses(ws, args, outbound_log, server_audio_chunks, t0_wall))

    for rec in records:
        if args.speed > 0:
            target = t0_wall + (rec["ts"] - t0_capture) / args.speed
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        msg = rec["data"]
        typ = msg.get("type", "?")

        if args.skip_cancel and typ == "response.cancel":
            print(f"  SKIP {typ}")
            continue
        if args.skip_session_update and typ == "session.update":
            print(f"  SKIP {typ}")
            continue
        if args.modalities and typ == "session.update":
            session_data = msg.get("session", {})
            if "output_modalities" in session_data:
                session_data["output_modalities"] = args.modalities.split(",")
                print(f"  OVERRIDE modalities -> {session_data['output_modalities']}")
        if "__VOICE__" in json.dumps(msg):
            print(f"  SKIP {typ} (no --voice)")
            continue

        audio = msg.get("audio", "")
        if audio:
            elapsed = time.monotonic() - t0_wall
            client_audio_chunks.append((elapsed, base64.b64decode(audio)))
        extra = f" ({len(audio)} chars b64)" if audio else ""
        elapsed = time.monotonic() - t0_wall
        print(f"  [{elapsed:7.3f}s] -> {typ}{extra}")

        await ws.send_str(json.dumps(msg))

    if args.wait > 0:
        print(f"All messages sent. Waiting {args.wait}s for responses...")
        await asyncio.sleep(args.wait)

    recv_task.cancel()
    outbound_log.close()
    await ws.close()
    await session.close()

    if client_audio_chunks or server_audio_chunks:
        client_buf = _build_timed_buffer(client_audio_chunks)
        server_buf = _build_timed_buffer(server_audio_chunks)
        _write_stereo_wav(args.audio_out, client_buf, server_buf)
        print(f"Wrote merged audio to {args.audio_out} "
              f"(client={len(client_buf)}B, server={len(server_buf)}B)")

    return 0


async def _print_responses(ws, args, outbound_log, server_audio_chunks, t0_wall):
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                typ = data.get("type", "?")
                detail = ""
                if typ in ("response.audio.delta", "response.output_audio.delta"):
                    delta = data.get("delta", "")
                    detail = f" ({len(delta)} chars b64)"
                    if delta:
                        elapsed = time.monotonic() - t0_wall
                        server_audio_chunks.append((elapsed, base64.b64decode(delta)))
                elif typ == "error":
                    detail = f" {data.get('error', data.get('message', ''))}"
                elif typ in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                    detail = f" {data.get('delta', '')!r}"
                print(f"  <-  {typ}{detail}")
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


_SESSION_DIR = os.path.dirname(__file__)
_VAD_SESSIONS = {
    "client": os.path.join(_SESSION_DIR, "livekit_session_client_vad.jsonl"),
    "semantic": os.path.join(_SESSION_DIR, "livekit_session_semantic_vad.jsonl"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad", choices=list(_VAD_SESSIONS), default="client",
                        help="VAD mode session to replay (default: client)")
    parser.add_argument("--inbound_log", help="Path to session JSONL (overrides --vad)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Replay speed multiplier (0 = no delay)",
    )
    parser.add_argument("--wait", type=float, default=10.0,
                        help="Seconds to wait for responses after last message")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Total session timeout in seconds (0 = no limit)")
    parser.add_argument("--skip-cancel", action="store_true",
                        help="Skip response.cancel messages")
    parser.add_argument("--skip-session-update", action="store_true",
                        help="Skip session.update messages")
    parser.add_argument("--outbound-log", default="/tmp/logs/replay_outbound.jsonl",
                        help="Path to write server responses (overwritten)")
    parser.add_argument("--dump-responses", action="store_true",
                        help="Print full JSON of server responses")
    parser.add_argument("--voice", action="store_true",
                        help="Upload reference-audio.wav and select it via session.update")
    parser.add_argument("--modalities", default=None,
                        help="Override output_modalities (e.g. 'text' or 'audio')")
    parser.add_argument("--audio-out", default="/tmp/logs/replay_merged.wav",
                        help="Path to write merged stereo WAV (left=client, right=server)")
    args = parser.parse_args()
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
