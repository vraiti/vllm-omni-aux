#!/usr/bin/env python3
"""Long-running /v1/live/sessions client that overflows the Stage 0 context.

Streams pregenerated utterances (``piper_utterances.py`` fixtures, 24 kHz PCM16)
in real time with silence in between for ``--duration`` seconds, and polls the
server's ``vllm_omni:live_context_trims`` counter. Exits non-zero unless at
least ``--min-trims`` trims happened, no ``error`` event arrived, the session
stayed open, and the model still produced audio after the first trim.

Writes ``events.jsonl``, ``summary.json`` and ``session.wav`` (left = user,
right = model, placed at arrival time) under ``--out``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np

RATE = 24_000
CHUNK_S = 0.1
CHUNK = int(RATE * CHUNK_S)
PLAN = ["capital", "population", "story", "hello", "weather", "french", "mmhm"]
TRIMS_RE = re.compile(r"^vllm_omni:live_context_trims(?:_total)?\{[^}]*\}\s+(\S+)$", re.M)


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wf:
        assert wf.getframerate() == RATE and wf.getnchannels() == 1 and wf.getsampwidth() == 2, path
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)


async def read_trims(http: aiohttp.ClientSession, url: str) -> float | None:
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            text = await resp.text()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None
    values = TRIMS_RE.findall(text)
    return sum(float(v) for v in values) if values else None


class Run:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.events: list[dict] = []
        self.out_audio: list[tuple[float, np.ndarray]] = []
        self.closed_at: float | None = None
        self.started = asyncio.Event()
        self.close_ack = asyncio.Event()

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 3)


async def receive(ws: aiohttp.ClientWebSocketResponse, run: Run, log) -> None:
    async for msg in ws:
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        event = json.loads(msg.data)
        t = run.now()
        kind = event.get("type", "")
        if kind == "session.output_audio.delta":
            run.out_audio.append((t, np.frombuffer(base64.b64decode(event["delta"]), dtype=np.int16)))
            record = {"t": t, "type": kind, "samples": len(run.out_audio[-1][1])}
        else:
            record = {"t": t, **event}
        run.events.append(record)
        log.write(json.dumps(record) + "\n")
        if kind == "session.started":
            run.started.set()
        elif kind == "session.closed":
            run.close_ack.set()
        elif kind == "error":
            print(f"[{t:7.1f}s] error: {event.get('error')}", flush=True)
        elif kind == "session.output_transcript.delta":
            print(f"[{t:7.1f}s] model: {event['delta']!r}", flush=True)
    run.closed_at = run.now()
    run.close_ack.set()


async def poll_trims(http: aiohttp.ClientSession, url: str, run: Run, timeline: list, stop: asyncio.Event) -> None:
    last = None
    while not stop.is_set():
        value = await read_trims(http, url)
        if value is not None and value != last:
            timeline.append({"t": run.now(), "trims": value})
            if last is not None:
                print(f"[{run.now():7.1f}s] context trims: {value:g}", flush=True)
            last = value
        try:
            await asyncio.wait_for(stop.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


async def main(args: argparse.Namespace) -> int:
    fixtures = {name: load_wav(args.fixtures / f"{name}.wav") for name in PLAN}
    args.out.mkdir(parents=True, exist_ok=True)
    run = Run()
    timeline: list[dict] = []
    utterances: list[dict] = []
    sent = np.zeros(int(RATE * (args.duration + 30)), dtype=np.int16)
    base = args.url.replace("ws://", "http://").split("/v1/")[0]
    metrics_url = f"{base}/metrics"

    async with aiohttp.ClientSession() as http, http.ws_connect(args.url, max_msg_size=0) as ws:
        with open(args.out / "events.jsonl", "w") as log:
            receiver = asyncio.create_task(receive(ws, run, log))
            stop_polling = asyncio.Event()
            poller = asyncio.create_task(poll_trims(http, metrics_url, run, timeline, stop_polling))
            await ws.send_json(
                {
                    "type": "session.start",
                    "session": {
                        "model": args.model,
                        "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": args.voice}},
                    },
                }
            )
            await asyncio.wait_for(run.started.wait(), timeout=60)
            stream_t0 = run.now()

            pos = 0
            index = 0
            next_utterance = int(RATE * args.lead_s)
            pending: np.ndarray | None = None
            deadline = time.monotonic() + args.duration
            tick = time.monotonic()
            while time.monotonic() < deadline and run.closed_at is None:
                if pending is None and pos >= next_utterance:
                    name = PLAN[index % len(PLAN)]
                    index += 1
                    pending = fixtures[name]
                    utterances.append({"wav": name, "start": run.now(), "end": run.now() + len(pending) / RATE})
                    next_utterance = pos + len(pending) + int(RATE * args.gap_s)
                if pending is not None:
                    chunk, pending = pending[:CHUNK], pending[CHUNK:]
                    if len(chunk) < CHUNK:
                        chunk = np.concatenate([chunk, np.zeros(CHUNK - len(chunk), dtype=np.int16)])
                    if pending.size == 0:
                        pending = None
                else:
                    chunk = np.zeros(CHUNK, dtype=np.int16)
                await ws.send_json({"type": "session.input_audio.append", "audio": base64.b64encode(chunk).decode()})
                sent[pos : pos + CHUNK] = chunk[: len(sent) - pos]
                pos += CHUNK
                tick += CHUNK_S
                await asyncio.sleep(max(0.0, tick - time.monotonic()))

            ended_early = run.closed_at is not None
            if not ended_early:
                await ws.send_json({"type": "session.close"})
                try:
                    await asyncio.wait_for(run.close_ack.wait(), timeout=15)
                except asyncio.TimeoutError:
                    pass
            await asyncio.sleep(3)
            stop_polling.set()
            await poller
            await ws.close()
            await receiver

    trims = timeline[-1]["trims"] - timeline[0]["trims"] if timeline else None
    first_trim_t = timeline[1]["t"] if len(timeline) > 1 else None
    audio_times = [t for t, _ in run.out_audio]
    for u in utterances:
        window_end = u["end"] + args.gap_s
        replies = [t for t in audio_times if u["start"] <= t < window_end]
        u["first_audio_latency_s"] = round(replies[0] - u["end"], 2) if replies else None
        u["after_first_trim"] = first_trim_t is not None and u["start"] > first_trim_t
    errors = [ev for ev in run.events if ev.get("type") == "error"]
    closed = [ev for ev in run.events if ev.get("type") == "session.closed"]
    transcript = "".join(ev["delta"] for ev in run.events if ev.get("type") == "session.output_transcript.delta")
    answered_after_trim = sum(1 for u in utterances if u["after_first_trim"] and u["first_audio_latency_s"] is not None)

    failures = []
    if trims is None:
        failures.append("vllm_omni:live_context_trims not found at /metrics")
    elif trims < args.min_trims:
        failures.append(f"{trims:g} context trims, expected at least {args.min_trims}")
    if errors:
        failures.append(f"{len(errors)} error event(s)")
    if ended_early:
        failures.append(f"socket closed by the server at {run.closed_at}s")
    if first_trim_t is not None and answered_after_trim == 0:
        failures.append("no audio reply to any utterance after the first trim")

    summary = {
        "duration_s": args.duration,
        "context_trims": trims,
        "trim_timeline": timeline,
        "utterances": utterances,
        "answered_after_first_trim": answered_after_trim,
        "output_audio_s": round(sum(len(a) for _, a in run.out_audio) / RATE, 2),
        "output_transcript": transcript,
        "errors": errors,
        "session_closed": closed,
        "failures": failures,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    end = max([len(sent)] + [int(RATE * (t - stream_t0)) + len(a) for t, a in run.out_audio])
    stereo = np.zeros((end, 2), dtype=np.int16)
    stereo[: len(sent), 0] = sent
    for t, audio in run.out_audio:
        start = max(0, int(RATE * (t - stream_t0)))
        stereo[start : start + len(audio), 1] = audio[: end - start]
    used = int(RATE * (run.now() - stream_t0))
    with wave.open(str(args.out / "session.wav"), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(stereo[:used].tobytes())

    print(json.dumps({k: v for k, v in summary.items() if k != "trim_timeline"}, indent=2))
    print(f"artifacts: {args.out}")
    print("PASS" if not failures else "FAIL: " + "; ".join(failures), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/live/sessions")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=360.0)
    parser.add_argument("--gap-s", type=float, default=12.0)
    parser.add_argument("--lead-s", type=float, default=1.0)
    parser.add_argument("--min-trims", type=int, default=1)
    sys.exit(asyncio.run(main(parser.parse_args())))
