#!/usr/bin/env python3
"""Accept WebSocket connections and write received messages to a JSONL file."""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone

import aiohttp
from aiohttp import web, WSMsgType


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    path = f"/tmp/logs/ws-sink-{ts}.jsonl"
    print(f"WS connect: {request.path} -> {path}")

    with open(path, "a") as f:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    data = msg.data
                entry = {"ts": time.time(), "data": data}
                f.write(json.dumps(entry) + "\n")
                f.flush()
            elif msg.type == WSMsgType.BINARY:
                entry = {"ts": time.time(), "binary_len": len(msg.data)}
                f.write(json.dumps(entry) + "\n")
                f.flush()
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break

    print(f"WS closed: {path}")
    return ws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int, help="Port to listen on")
    args = parser.parse_args()

    app = web.Application()
    app.router.add_route("GET", "/{path:.*}", handle_ws)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)
    print(f"Listening on 0.0.0.0:{args.port}")


if __name__ == "__main__":
    main()
