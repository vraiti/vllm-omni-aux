#!/usr/bin/env python3
"""Serve any diffusers image generation model with a minimal HTTP API."""

import argparse
import base64
import io
import json
import time

import torch
from diffusers import DiffusionPipeline
from http.server import HTTPServer, BaseHTTPRequestHandler


def load_pipeline(model: str, device_ids: list[int], offload: str, fp8: bool = False) -> DiffusionPipeline:
    kwargs = {"torch_dtype": torch.bfloat16}
    if offload == "none" and len(device_ids) > 1:
        kwargs["device_map"] = "balanced"

    pipe = DiffusionPipeline.from_pretrained(model, **kwargs)

    if fp8 and hasattr(pipe, "transformer"):
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
        )

    if "device_map" not in kwargs:
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload(gpu_id=device_ids[0])
        elif offload == "none":
            pipe = pipe.to(f"cuda:{device_ids[0]}")
        else:
            pipe.enable_model_cpu_offload(gpu_id=device_ids[0])
    return pipe


class Handler(BaseHTTPRequestHandler):
    pipe: DiffusionPipeline

    def do_POST(self):
        if self.path != "/v1/images/generations":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        prompt = body.get("prompt", "")
        size = body.get("size", "1024x1024")
        w, h = (int(x) for x in size.split("x"))
        extra = {k: v for k, v in body.items() if k not in ("prompt", "size", "model")}

        t0 = time.perf_counter()
        image = self.pipe(prompt=prompt, width=w, height=h, **extra).images[0]
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
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--devices", default="0", help="CUDA device IDs (comma-separated)")
    parser.add_argument("--offload", choices=["model", "sequential", "none"], default="model",
                        help="CPU offload strategy (default: model)")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 layerwise casting on transformer")
    args = parser.parse_args()

    device_ids = [int(d) for d in args.devices.split(",")]
    print(f"Loading {args.model} on devices {device_ids}, offload={args.offload}, fp8={args.fp8}")
    pipe = load_pipeline(args.model, device_ids, args.offload, fp8=args.fp8)
    Handler.pipe = pipe

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Serving on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
