#!/usr/bin/env python3
"""WS-disconnect repro harness: drives the REAL livekit-plugins-openai
RealtimeModel/RealtimeSession client against the mock server
(ws_repro_mock_server.py), with no AgentSession/room/audio hardware
involved -- just synthetic audio pushed continuously plus a couple of
early generate_reply() calls to mimic a real conversation opening. Runs
ONCE for a fixed watch period, then exits and reports pass/fail -- no
internal repeat loop.

If this reproduces the same abrupt "connection closed unexpectedly" /
ws_conn.closed=True-with-no-close-frame pattern we've seen in production,
that clears vLLM-Omni's server entirely: the bug (if any) is in the client
library + network path, independent of any application code.

Run from the agent's own venv so it picks up the already-instrumented
(diagnostic-logging) copy of realtime_model.py:

    cd voice-assistant-with-vllm-omni/agent
    uv run python3 /path/to/ws_repro_client.py --base-url http://<mock-ip>:8000/v1 --watch 300
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np
from livekit import rtc
from livekit.plugins.openai.realtime import RealtimeModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ws_repro_client")

SAMPLE_RATE = 24000
FRAME_MS = 100
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000

disconnect_seen = False


class _DisconnectWatcher(logging.Handler):
    """Flags a real reconnect event logged by realtime_model.py itself,
    rather than inferring it from our own generate_reply() calls."""

    def emit(self, record: logging.LogRecord) -> None:
        global disconnect_seen
        msg = record.getMessage()
        if "connection closed unexpectedly" in msg or "_run_ws finally" in msg:
            disconnect_seen = True


async def audio_pump(session, stop: asyncio.Event) -> None:
    rng = np.random.default_rng(0)
    while not stop.is_set():
        samples = rng.integers(-200, 200, size=SAMPLES_PER_FRAME, dtype=np.int16)
        frame = rtc.AudioFrame(samples.tobytes(), SAMPLE_RATE, 1, SAMPLES_PER_FRAME)
        session.push_audio(frame)
        await asyncio.sleep(FRAME_MS / 1000)


async def opening_turns(session, n: int, gap_s: float) -> None:
    """A handful of turns right at the start, mimicking a real conversation
    opening -- not an ongoing repeating loop for the rest of the run."""
    for turn_n in range(1, n + 1):
        await asyncio.sleep(gap_s)
        session.commit_audio()
        try:
            handle = session.generate_reply()
            await asyncio.wait_for(handle, timeout=10.0)
            logger.info("[repro] opening turn %d/%d: reply generated ok", turn_n, n)
        except Exception as e:
            logger.warning("[repro] opening turn %d/%d: generate_reply failed: %r", turn_n, n, e)


async def status_loop(start: float, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(30)
        logger.info("[repro] still watching, uptime=%.0fs, disconnect_seen=%s", time.monotonic() - start, disconnect_seen)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="e.g. http://<mock-ip>:8000/v1")
    parser.add_argument("--watch", type=float, default=300.0, help="total seconds to run, once")
    parser.add_argument("--opening-turns", type=int, default=2, help="turns to generate at the start")
    args = parser.parse_args()

    logging.getLogger("livekit.plugins.openai").addHandler(_DisconnectWatcher())

    model = RealtimeModel(
        base_url=args.base_url,
        model="mock-model",
        api_key="unused",
        turn_detection=None,
    )
    session = model.session()

    stop = asyncio.Event()
    start = time.monotonic()
    pump_task = asyncio.create_task(audio_pump(session, stop))
    status_task = asyncio.create_task(status_loop(start, stop))

    try:
        await opening_turns(session, args.opening_turns, gap_s=15.0)
        remaining = args.watch - (time.monotonic() - start)
        if remaining > 0:
            logger.info("[repro] opening turns done, watching quietly for %.0fs more", remaining)
            await asyncio.sleep(remaining)
    finally:
        stop.set()
        pump_task.cancel()
        status_task.cancel()
        await asyncio.gather(pump_task, status_task, return_exceptions=True)
        await model.aclose()
        uptime = time.monotonic() - start
        if disconnect_seen:
            logger.warning("[repro] RESULT: disconnect observed during the %.0fs run", uptime)
        else:
            logger.info("[repro] RESULT: no disconnect observed during the %.0fs run", uptime)


if __name__ == "__main__":
    asyncio.run(main())
