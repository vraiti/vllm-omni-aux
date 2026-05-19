#!/usr/bin/env python3
"""Generate images from prompts.txt and save outputs as PNGs.

Works with both vLLM-Omni and the diffusers baseline since they
share the same /v1/images/generations API shape.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request


def generate(url: str, prompt: str, size: str) -> tuple[bytes, float]:
    body = json.dumps({"prompt": prompt, "size": size}).encode()
    req = urllib.request.Request(
        f"{url}/v1/images/generations",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    b64 = data["data"][0]["b64_json"]
    return base64.b64decode(b64), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--prompts", default="prompts.txt")
    parser.add_argument("--output-dir", required=True, help="Directory to save PNGs")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--tag", default="", help="Prefix for output filenames")
    args = parser.parse_args()

    prompts_path = args.prompts
    if not os.path.isabs(prompts_path):
        prompts_path = os.path.join(os.path.dirname(__file__), prompts_path)
    with open(prompts_path) as f:
        prompts = [line.strip() for line in f if line.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{args.tag}_" if args.tag else ""
    results = []

    for i, prompt in enumerate(prompts):
        fname = f"{prefix}{i:02d}.png"
        out_path = os.path.join(args.output_dir, fname)
        print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...", flush=True)
        img_bytes, elapsed = generate(args.url, prompt, args.size)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        results.append({"prompt": prompt, "file": fname, "latency_s": round(elapsed, 3)})
        print(f"  -> {fname} ({elapsed:.1f}s)", flush=True)

    summary = {
        "url": args.url,
        "size": args.size,
        "tag": args.tag,
        "results": results,
    }
    summary_path = os.path.join(args.output_dir, f"{prefix}summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
