#!/usr/bin/env python3
"""Replay client WebSocket messages from a proxy JSONL capture."""

import argparse
import asyncio
import json
import os
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

    outbound_log = open(args.outbound_log, "w")
    recv_task = asyncio.create_task(_print_responses(ws, args, outbound_log))

    t0_capture = records[0]["ts"]
    t0_wall = time.monotonic()

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
        if "__VOICE__" in json.dumps(msg):
            print(f"  SKIP {typ} (no --voice)")
            continue

        audio = msg.get("audio", "")
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
    return 0


async def _print_responses(ws, args, outbound_log):
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                typ = data.get("type", "?")
                detail = ""
                if typ == "response.audio.delta":
                    audio = data.get("delta", "")
                    detail = f" ({len(audio)} chars b64)"
                elif typ == "error":
                    detail = f" {data.get('error', data.get('message', ''))}"
                elif typ == "response.audio_transcript.delta":
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
