#!/usr/bin/env python3
"""Minimal mock /v1/realtime server for isolating the WS-disconnect repro.

Implements just enough of the OpenAI Realtime protocol to drive the real
livekit-plugins-openai RealtimeModel/RealtimeSession client through a normal
session lifecycle -- session.update, input_audio_buffer.append/commit,
response.create -- with entirely scripted/synthetic responses. No model, no
GPU: the goal is to isolate whether the client-library + network path alone
reproduces the disconnect, independent of vLLM-Omni's own server code.

Usage:
    uvicorn ws_repro_mock_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ws_repro_mock_server")

app = FastAPI()

SAMPLE_RATE = 24000


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _synthetic_pcm16_b64(num_samples: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    samples = rng.integers(-500, 500, size=num_samples, dtype=np.int16)
    return base64.b64encode(samples.tobytes()).decode()


async def _send(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload))


async def _run_scripted_response(ws: WebSocket, response_id: str) -> None:
    """Streams a canned response matching the real server's rough event
    shape and audio-chunk sizes observed in production captures."""
    item_id = _gen_id("item")
    output_index = 0
    content_index = 0

    await _send(
        ws,
        {
            "event_id": _gen_id("evt"),
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        },
    )
    await _send(
        ws,
        {
            "event_id": _gen_id("evt"),
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": {"type": "audio", "audio": "", "text": "", "transcript": ""},
        },
    )

    transcript_words = [
        "This",
        "is",
        "a",
        "scripted",
        "reply",
        "from",
        "the",
        "mock",
        "server",
        "used",
        "to",
        "isolate",
        "the",
        "websocket",
        "disconnect.",
    ]
    full_transcript = ""
    for word in transcript_words:
        delta = (" " if full_transcript else "") + word
        full_transcript += delta
        await _send(
            ws,
            {
                "event_id": _gen_id("evt"),
                "type": "response.output_audio_transcript.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": delta,
            },
        )
        await asyncio.sleep(0.03)

    # ~2s of audio total, first chunk large (matches the first delta being
    # much bigger than the rest in real captures), then smaller chunks.
    chunk_sizes = [47445, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
    seed_base = int(time.time() * 1000) % 100000
    for i, n in enumerate(chunk_sizes):
        b64 = _synthetic_pcm16_b64(n, seed_base + i)
        await _send(
            ws,
            {
                "event_id": _gen_id("evt"),
                "type": "response.output_audio.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": b64,
            },
        )
        await asyncio.sleep(0.05)

    await _send(
        ws,
        {
            "event_id": _gen_id("evt"),
            "type": "response.output_audio.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
        },
    )
    await _send(
        ws,
        {
            "event_id": _gen_id("evt"),
            "type": "response.output_audio_transcript.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "transcript": full_transcript,
        },
    )
    await _send(
        ws,
        {
            "event_id": _gen_id("evt"),
            "type": "response.content_part.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": {"type": "audio", "text": "", "transcript": full_transcript},
        },
    )
    item = {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_audio", "transcript": full_transcript}],
    }
    await _send(
        ws,
        {
            "event_id": _gen_id("evt"),
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": output_index,
            "item": item,
        },
    )
    await _send(
        ws,
        {
            "event_id": _gen_id("evt"),
            "type": "response.done",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "completed",
                "output": [item],
                "usage": {"total_tokens": 42, "input_tokens": 20, "output_tokens": 22},
            },
        },
    )


@app.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket) -> None:
    await ws.accept()
    conn_id = _gen_id("conn")
    session_id = _gen_id("sess")
    conversation_id = _gen_id("conv")
    connected_at = time.monotonic()
    logger.info("[%s] connection accepted", conn_id)

    await _send(
        ws,
        {
            "event_id": _gen_id("evt"),
            "type": "session.created",
            "session": {
                "id": session_id,
                "object": "realtime.session",
                "type": "realtime",
                "model": "mock-model",
                "output_modalities": ["audio"],
                "max_output_tokens": "inf",
            },
        },
    )
    await _send(
        ws,
        {
            "event_id": _gen_id("evt"),
            "type": "conversation.created",
            "conversation": {"id": conversation_id, "object": "realtime.conversation"},
        },
    )

    response_tasks: list[asyncio.Task] = []
    try:
        while True:
            text = await ws.receive_text()
            event = json.loads(text)
            etype = event.get("type")

            if etype == "session.update":
                await _send(
                    ws,
                    {
                        "event_id": _gen_id("evt"),
                        "type": "session.updated",
                        "session": {
                            "id": session_id,
                            "object": "realtime.session",
                            "type": "realtime",
                            **event.get("session", {}),
                        },
                    },
                )
            elif etype == "input_audio_buffer.commit":
                item_id = _gen_id("item")
                await _send(
                    ws,
                    {
                        "event_id": _gen_id("evt"),
                        "type": "input_audio_buffer.committed",
                        "item_id": item_id,
                    },
                )
                await _send(
                    ws,
                    {
                        "event_id": _gen_id("evt"),
                        "type": "conversation.item.created",
                        "item": {
                            "id": item_id,
                            "object": "realtime.item",
                            "type": "message",
                            "role": "user",
                            "status": "completed",
                            "content": [{"type": "input_audio", "audio": ""}],
                        },
                    },
                )
            elif etype == "response.create":
                response_id = _gen_id("resp")
                await _send(
                    ws,
                    {
                        "event_id": _gen_id("evt"),
                        "type": "response.created",
                        "response": {
                            "id": response_id,
                            "object": "realtime.response",
                            "status": "in_progress",
                            "output": [],
                            "metadata": (event.get("response") or {}).get("metadata"),
                        },
                    },
                )
                task = asyncio.create_task(_run_scripted_response(ws, response_id))
                response_tasks.append(task)
            # input_audio_buffer.append, response.cancel, everything else: no-op,
            # matching how a real server doesn't ack every single append.
    except WebSocketDisconnect as e:
        elapsed = time.monotonic() - connected_at
        logger.warning("[%s] WebSocketDisconnect after %.1fs: code=%s reason=%r", conn_id, elapsed, e.code, e.reason)
    except Exception:
        elapsed = time.monotonic() - connected_at
        logger.exception("[%s] connection error after %.1fs", conn_id, elapsed)
    finally:
        for t in response_tasks:
            if not t.done():
                t.cancel()
        logger.info("[%s] connection closed (alive %.1fs)", conn_id, time.monotonic() - connected_at)
