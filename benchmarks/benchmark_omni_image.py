#!/usr/bin/env python3
"""Benchmark FLUX.2-dev image generation served by vLLM-Omni."""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = "black-forest-labs/FLUX.2-dev"
DEFAULT_PROMPT = "a red cat sitting on a windowsill"


def send_request(url: str, model: str, prompt: str, extra_params: dict, save_image: bool = False) -> dict:
    payload = {"model": model, "prompt": prompt, **extra_params}
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/images/generations",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )
        latency = time.perf_counter() - t0
        result = {"latency": latency, "status": resp.status_code}
        if save_image and resp.status_code == 200:
            data = resp.json().get("data", [])
            if data and "b64_json" in data[0]:
                result["image_bytes"] = base64.b64decode(data[0]["b64_json"])
        return result
    except Exception as e:
        latency = time.perf_counter() - t0
        return {"latency": latency, "status": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark FLUX.2-dev on vLLM-Omni")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for image generation")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup requests")
    parser.add_argument("--requests", type=int, default=5, help="Number of benchmark requests")
    parser.add_argument("--size", default="1024x1024", help="Image size WxH")
    parser.add_argument("--extra", type=str, default=None, help="Extra OpenAI params as JSON string")
    parser.add_argument("--output", default="benchmark_flux2.json", help="Output JSON path")
    parser.add_argument("--save-images", default=None, metavar="DIR", help="Save generated images to DIR")
    args = parser.parse_args()

    url = args.url.rstrip("/")
    extra_params = {"size": args.size}
    if args.extra:
        extra_params.update(json.loads(args.extra))

    try:
        r = requests.get(f"{url}/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Server not reachable at {url}: {e}", file=sys.stderr)
        sys.exit(1)

    for i in range(args.warmup):
        sys.stdout.write(f"  warmup {i + 1}/{args.warmup} ... ")
        sys.stdout.flush()
        r = send_request(url, args.model, args.prompt, extra_params)
        print(f"{r['status']}  {r['latency']:.2f}s")

    save_dir = args.save_images
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    index = {}
    for i in range(args.requests):
        sys.stdout.write(f"  request {i + 1}/{args.requests} ... ")
        sys.stdout.flush()
        r = send_request(url, args.model, args.prompt, extra_params, save_image=bool(save_dir))
        results.append({"latency": r["latency"], "status": r["status"]})
        print(f"{r['status']}  {r['latency']:.2f}s")
        if save_dir and "image_bytes" in r:
            fname = f"{i + 1}.png"
            with open(os.path.join(save_dir, fname), "wb") as f:
                f.write(r["image_bytes"])
            index[fname] = args.prompt

    output = {
        "config": {
            "url": url,
            "model": args.model,
            "prompt": args.prompt,
            "warmup": args.warmup,
            "requests": args.requests,
            "size": args.size,
            "extra": args.extra,
            "output": args.output,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    if save_dir and index:
        index_path = os.path.join(save_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Images saved to {save_dir}/")


if __name__ == "__main__":
    main()
