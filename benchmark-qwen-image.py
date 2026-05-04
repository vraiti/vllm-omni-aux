#!/usr/bin/env python3
"""
Benchmark Qwen-Image-2512 text-to-image generation.

Measures: E2E latency, image size, and throughput (images/min).

Usage:
  python benchmark-qwen-image.py --url http://localhost:8000 --samples 5
  python benchmark-qwen-image.py --url http://localhost:8000 --samples 3 --steps 30 --size 512x512
"""

import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import requests

DEFAULT_MODEL = "Qwen/Qwen-Image-2512"

PROMPTS = [
    "A photorealistic close-up of a steaming cup of coffee on a wooden table, morning light streaming through a window",
    "An oil painting of a medieval castle on a cliff overlooking the sea at sunset",
    "A cyberpunk cityscape at night with neon signs reflecting on wet streets",
    "A detailed pencil sketch of an old oak tree in a meadow with mountains in the background",
    "A watercolor illustration of a cozy bookshop interior with a cat sleeping on a stack of books",
    "A macro photograph of dewdrops on a spider web with bokeh background",
    "An isometric pixel art scene of a busy Japanese ramen shop",
    "A studio portrait of an astronaut holding a bouquet of wildflowers, dramatic lighting",
    "A drone aerial view of terraced rice paddies during golden hour",
    "A vintage travel poster for Mars featuring retro 1950s typography and a rocket ship",
]


def send_request(
    url: str,
    prompt: str,
    model: str,
    size: str,
    steps: int,
    cfg_scale: float,
    seed: int | None,
) -> dict:
    payload = {
        "prompt": prompt,
        "response_format": "b64_json",
        "n": 1,
        "size": size,
        "num_inference_steps": steps,
        "true_cfg_scale": cfg_scale,
    }
    if seed is not None:
        payload["seed"] = seed

    t_start = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/images/generations",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )
        resp.raise_for_status()
        t_end = time.perf_counter()
        data = resp.json()

        items = data.get("data", [])
        if not items or "b64_json" not in items[0]:
            return {
                "success": False,
                "error": f"unexpected response: {json.dumps(data)[:200]}",
                "latency_s": t_end - t_start,
            }

        img_bytes = base64.b64decode(items[0]["b64_json"])
        return {
            "success": True,
            "latency_s": t_end - t_start,
            "image_bytes": len(img_bytes),
        }

    except requests.exceptions.HTTPError as e:
        t_end = time.perf_counter()
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        return {"success": False, "error": f"{e} | {body}", "latency_s": t_end - t_start}
    except Exception as e:
        t_end = time.perf_counter()
        return {"success": False, "error": str(e), "latency_s": t_end - t_start}


def run_benchmark(
    url: str,
    prompts: list[str],
    model: str,
    size: str,
    steps: int,
    cfg_scale: float,
    seed: int | None,
    warmup: int,
) -> dict:
    n = len(prompts)
    print(f"\n{'='*60}")
    print(f"  Qwen-Image-2512 Benchmark")
    print(f"  {n} requests, size={size}, steps={steps}, cfg={cfg_scale}")
    print(f"  warmup: {warmup}")
    print(f"{'='*60}")

    for w in range(warmup):
        sys.stdout.write(f"  [warmup {w+1}/{warmup}] ")
        sys.stdout.flush()
        r = send_request(url, prompts[w % n], model, size, steps, cfg_scale, seed)
        if r["success"]:
            print(f"OK  {r['latency_s']:.2f}s  {r['image_bytes']/1024:.0f}KB (discarded)")
        else:
            print(f"FAIL  {r['error'][:80]} (discarded)")

    results = []
    for i, prompt in enumerate(prompts):
        sys.stdout.write(f"  [{i+1}/{n}] ")
        sys.stdout.flush()
        r = send_request(url, prompt, model, size, steps, cfg_scale, seed)
        r["prompt"] = prompt
        results.append(r)
        if r["success"]:
            print(f"OK  {r['latency_s']:.2f}s  {r['image_bytes']/1024:.0f}KB")
        else:
            print(f"FAIL  {r['error'][:80]}")

    successes = [r for r in results if r["success"]]
    failures = len(results) - len(successes)

    if not successes:
        print("\n  All requests failed.")
        return {"total": n, "failures": failures, "error": "all requests failed", "requests": results}

    latencies = [r["latency_s"] for r in successes]
    sizes_kb = [r["image_bytes"] / 1024 for r in successes]

    def pct(vals, p):
        if len(vals) < 2:
            return vals[0] if vals else None
        return sorted(vals)[int(len(vals) * p)]

    summary = {
        "total": n,
        "successes": len(successes),
        "failures": failures,
        "config": {
            "size": size,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
        },
        "latency": {
            "mean_s": statistics.mean(latencies),
            "median_s": statistics.median(latencies),
            "p95_s": pct(latencies, 0.95),
            "min_s": min(latencies),
            "max_s": max(latencies),
        },
        "image_size_kb": {
            "mean": statistics.mean(sizes_kb),
            "median": statistics.median(sizes_kb),
        },
        "throughput_img_per_min": 60.0 / statistics.mean(latencies),
        "requests": results,
    }

    print(f"\n  Latency:      mean={summary['latency']['mean_s']:.2f}s  "
          f"median={summary['latency']['median_s']:.2f}s  "
          f"p95={summary['latency']['p95_s']:.2f}s")
    print(f"  Range:        min={summary['latency']['min_s']:.2f}s  "
          f"max={summary['latency']['max_s']:.2f}s")
    print(f"  Image size:   mean={summary['image_size_kb']['mean']:.0f}KB  "
          f"median={summary['image_size_kb']['median']:.0f}KB")
    print(f"  Throughput:   {summary['throughput_img_per_min']:.2f} img/min")
    print(f"  Failures:     {failures}/{n}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen-Image-2512 text-to-image generation")
    parser.add_argument("--url", required=True, help="Base URL of the vLLM-Omni server")
    parser.add_argument("--samples", type=int, default=5, help="Number of images to generate")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests (not recorded)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--size", default="1024x1024", help="Image size WxH (default: 1024x1024)")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion inference steps")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--output", default="benchmark_qwen_image.json", help="Output JSON file")
    args = parser.parse_args()

    url = args.url.rstrip("/")

    try:
        r = requests.get(f"{url}/health", timeout=10)
        r.raise_for_status()
        print(f"Server healthy: {url}")
    except Exception as e:
        print(f"Server not reachable at {url}: {e}")
        sys.exit(1)

    prompts = (PROMPTS * ((args.samples // len(PROMPTS)) + 1))[:args.samples]

    summary = run_benchmark(
        url=url,
        prompts=prompts,
        model=args.model,
        size=args.size,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        warmup=args.warmup,
    )

    output_path = Path(__file__).parent / args.output
    save_data = {
        "summary": {k: v for k, v in summary.items() if k != "requests"},
        "config": {"url": url, "samples": args.samples, "model": args.model},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    full_path = output_path.with_suffix(".full.json")
    with open(full_path, "w") as f:
        json.dump({"results": summary, "config": {"url": url, "samples": args.samples, "model": args.model}}, f, indent=2)
    print(f"Full results saved to {full_path}")


if __name__ == "__main__":
    main()
