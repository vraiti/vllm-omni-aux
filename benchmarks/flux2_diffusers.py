#!/usr/bin/env python3
"""Serve FLUX.2-dev using diffusers with a minimal HTTP API."""

import argparse
import base64
import io
import json
import time

import torch
from diffusers import FluxPipeline
from http.server import HTTPServer, BaseHTTPRequestHandler


def load_pipeline(device_ids: list[int], offload: str | None) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        torch_dtype=torch.bfloat16,
    )
    if offload == "model":
        pipe.enable_model_cpu_offload(gpu_id=device_ids[0])
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=device_ids[0])
    else:
        pipe = pipe.to(f"cuda:{device_ids[0]}")
    return pipe


class Handler(BaseHTTPRequestHandler):
    pipe: FluxPipeline

    def do_POST(self):
        if self.path != "/v1/images/generations":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        prompt = body.get("prompt", "")
        size = body.get("size", "1024x1024")
        w, h = (int(x) for x in size.split("x"))
        steps = body.get("num_inference_steps", 28)

        t0 = time.perf_counter()
        image = self.pipe(
            prompt=prompt,
            width=w,
            height=h,
            num_inference_steps=steps,
            guidance_scale=3.5,
        ).images[0]
        elapsed = time.perf_counter() - t0

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = json.dumps({
            "created": int(time.time()),
            "size": size,
            "inference_time_s": round(elapsed, 3),
            "data": [{"b64_json": b64}],
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--devices", default="0,1", help="CUDA device IDs")
    parser.add_argument("--offload", choices=["model", "sequential"], default=None,
                        help="CPU offload strategy")
    args = parser.parse_args()

    device_ids = [int(d) for d in args.devices.split(",")]
    print(f"Loading FLUX.2-dev on devices {device_ids}, offload={args.offload}")
    pipe = load_pipeline(device_ids, args.offload)
    Handler.pipe = pipe

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Serving on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
