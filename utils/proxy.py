#!/usr/bin/env python3
import argparse
import asyncio
import json
import time

import aiohttp
from aiohttp import web, WSMsgType


async def proxy_ws(request: web.Request) -> web.WebSocketResponse:
    client_ws = web.WebSocketResponse()
    await client_ws.prepare(request)

    upstream_url = request.app["upstream_url"] + request.path_qs
    inbound_log = open(request.app["inbound_log"], "a")
    outbound_log = open(request.app["outbound_log"], "a")

    session = aiohttp.ClientSession()
    try:
        upstream_ws = await session.ws_connect(upstream_url)
    except Exception as e:
        print(f"Failed to connect to upstream: {e}")
        inbound_log.close()
        outbound_log.close()
        await session.close()
        await client_ws.close(message=str(e).encode())
        return client_ws

    async def client_to_upstream():
        async for msg in client_ws:
            if msg.type == WSMsgType.TEXT:
                record = {"ts": time.time(), "data": json.loads(msg.data)}
                inbound_log.write(json.dumps(record) + "\n")
                inbound_log.flush()
                await upstream_ws.send_str(msg.data)
            elif msg.type == WSMsgType.BINARY:
                await upstream_ws.send_bytes(msg.data)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
        await upstream_ws.close()

    async def upstream_to_client():
        async for msg in upstream_ws:
            if msg.type == WSMsgType.TEXT:
                record = {"ts": time.time(), "data": json.loads(msg.data)}
                outbound_log.write(json.dumps(record) + "\n")
                outbound_log.flush()
                await client_ws.send_str(msg.data)
            elif msg.type == WSMsgType.BINARY:
                await client_ws.send_bytes(msg.data)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                break
        await client_ws.close()

    try:
        await asyncio.gather(client_to_upstream(), upstream_to_client())
    finally:
        inbound_log.close()
        outbound_log.close()
        await session.close()

    return client_ws


async def proxy_http(request: web.Request) -> web.Response:
    upstream_url = request.app["upstream_url"] + request.path_qs
    async with aiohttp.ClientSession() as session:
        async with session.request(
            request.method, upstream_url,
            headers=request.headers, data=await request.read(),
        ) as resp:
            body = await resp.read()
            return web.Response(status=resp.status, headers=resp.headers, body=body)


async def handler(request: web.Request) -> web.WebSocketResponse | web.Response:
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await proxy_ws(request)
    return await proxy_http(request)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=int, default=9090)
    parser.add_argument("--upstream", type=int, default=8000)
    parser.add_argument("--inbound-log", default="/tmp/logs/proxy_inbound.jsonl")
    parser.add_argument("--outbound-log", default="/tmp/logs/proxy_outbound.jsonl")
    args = parser.parse_args()

    app = web.Application()
    app["upstream_url"] = f"http://127.0.0.1:{args.upstream}"
    app["inbound_log"] = args.inbound_log
    app["outbound_log"] = args.outbound_log
    app.router.add_route("*", "/{path:.*}", handler)

    print(f"Proxying 0.0.0.0:{args.listen} -> 127.0.0.1:{args.upstream}")
    print(f"  inbound log:  {args.inbound_log}")
    print(f"  outbound log: {args.outbound_log}")
    web.run_app(app, host="0.0.0.0", port=args.listen, print=None)


if __name__ == "__main__":
    main()
