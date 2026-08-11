#!/usr/bin/env python3
"""Accept WebSocket connections and write received messages to a JSONL file.

Proxies /health and /v1/models to the vLLM server on localhost:8000.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import aiohttp
from aiohttp import web, WSMsgType

UPSTREAM = "http://localhost:8000"
PROXY_PATHS = {"/health", "/v1/models"}


async def handle_proxy(request: web.Request) -> web.StreamResponse:
    url = f"{UPSTREAM}{request.path}"
    async with aiohttp.ClientSession() as client:
        async with client.request(
            request.method,
            url,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            data=await request.read(),
        ) as resp:
            response = web.StreamResponse(status=resp.status, headers=resp.headers)
            await response.prepare(request)
            async for chunk in resp.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response


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


async def route_handler(request: web.Request):
    if request.path in PROXY_PATHS:
        return await handle_proxy(request)
    return await handle_ws(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int, help="Port to listen on")
    args = parser.parse_args()

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", route_handler)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)
    print(f"Listening on 0.0.0.0:{args.port}")


if __name__ == "__main__":
    main()
