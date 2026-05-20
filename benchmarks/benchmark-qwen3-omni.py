#!/usr/bin/env python3
"""
Benchmark Qwen3-Omni-Instruct across 5 modality combinations:
  1. Text -> Text+Audio
  2. Image -> Text+Audio
  3. Audio -> Text+Audio
  4. Video -> Text+Audio
  5. Text -> Audio (TTS)

Measures: TTFT, TTLT (time to last text token), TTFA, E2E latency,
text tok/s (isolated Thinker throughput), overall tok/s.

Usage:
  python benchmark-instruct.py --url http://localhost:8000 \
                                --samples 3 --output benchmark_instruct.json
"""

import argparse
import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data-samples"
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL = DEFAULT_MODEL


def b64_encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_text_samples(n: int) -> list[str]:
    text_dir = DATA_DIR / "text"
    samples = []
    for f in sorted(text_dir.glob("sample_*.json")):
        with open(f) as fh:
            prompt = json.load(fh)["prompt"]
        samples.append(prompt)
        if len(samples) >= n:
            break
    return samples


def load_image_paths(n: int) -> list[str]:
    img_dir = DATA_DIR / "images"
    return [str(p) for p in sorted(img_dir.glob("image_*.jpg"))[:n]]


def load_audio_paths(n: int) -> list[str]:
    audio_dir = DATA_DIR / "audio"
    return [str(p) for p in sorted(audio_dir.glob("*.wav"))[:n]]


def load_video_paths(n: int) -> list[str]:
    video_dir = DATA_DIR / "video"
    return [str(p) for p in sorted(video_dir.glob("clip_*.mp4"))[:n]]


def build_text_to_text_audio(prompt: str) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.7,
    }


def build_image_to_text_audio(prompt: str, image_path: str) -> dict:
    b64 = b64_encode_file(image_path)
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }


def build_audio_to_text_audio(prompt: str, audio_path: str) -> dict:
    b64 = b64_encode_file(audio_path)
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": "wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }


def build_video_to_text_audio(prompt: str, video_path: str) -> dict:
    b64 = b64_encode_file(video_path)
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }


def build_tts(prompt: str) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Read the following aloud: {prompt}"}],
        "max_tokens": 256,
        "temperature": 0.7,
    }


def send_request(url: str, payload: dict) -> dict:
    """Send a streaming chat completion request and measure timing milestones."""
    endpoint = f"{url}/v1/chat/completions"
    streaming_payload = {**payload, "stream": True}

    t_start = time.perf_counter()
    ttft = None   # time to first text token
    ttlt = None   # time to last text token
    ttfa = None   # time to first audio chunk
    text_tokens = 0
    completion_tokens = 0
    prompt_tokens = 0
    has_audio = False
    text_content = []

    try:
        resp = requests.post(
            endpoint,
            json=streaming_payload,
            headers={"Content-Type": "application/json"},
            timeout=300,
            stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                usage = chunk["usage"]
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)

            modality = chunk.get("modality", "text")
            choices = chunk.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {})
                content = delta.get("content")

                if modality == "audio":
                    if content and ttfa is None:
                        ttfa = time.perf_counter() - t_start
                        has_audio = True
                else:
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        ttlt = time.perf_counter() - t_start
                        text_tokens += 1
                        text_content.append(content)

        t_end = time.perf_counter()

    except requests.exceptions.HTTPError as e:
        t_end = time.perf_counter()
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        return {
            "success": False,
            "error": f"{e} | {body}",
            "latency_s": t_end - t_start,
        }
    except Exception as e:
        t_end = time.perf_counter()
        return {
            "success": False,
            "error": str(e),
            "latency_s": t_end - t_start,
        }

    latency = t_end - t_start

    # Text-only throughput: text tokens / time spent generating text
    text_duration = (ttlt - ttft) if (ttft is not None and ttlt is not None and ttlt > ttft) else None
    text_tok_per_sec = text_tokens / text_duration if text_duration and text_tokens > 0 else 0

    # Overall throughput
    if completion_tokens == 0 and text_content:
        completion_tokens = len(text_content)
    overall_tok_per_sec = completion_tokens / latency if latency > 0 and completion_tokens > 0 else 0

    return {
        "success": True,
        "latency_s": latency,
        "ttft_s": ttft,
        "ttlt_s": ttlt,
        "ttfa_s": ttfa,
        "text_tokens": text_tokens,
        "text_duration_s": text_duration,
        "text_tok_per_sec": text_tok_per_sec,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "overall_tok_per_sec": overall_tok_per_sec,
        "text_output_len": len("".join(text_content)),
        "has_audio": has_audio,
    }


def fmt_metric(val, suffix="s"):
    if val is None:
        return "n/a"
    return f"{val:.2f}{suffix}"


def run_benchmark(name: str, url: str, payloads: list[dict], warmup: int = 0) -> dict:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {len(payloads)} requests (warmup: {warmup})")
    print(f"{'='*60}")

    for w in range(warmup):
        sys.stdout.write(f"  [warmup {w+1}/{warmup}] ")
        sys.stdout.flush()
        r = send_request(url, payloads[w % len(payloads)])
        if r["success"]:
            print(f"OK  {r['latency_s']:.2f}s (discarded)")
        else:
            print(f"FAIL  {r['error'][:80]} (discarded)")

    results = []
    for i, payload in enumerate(payloads):
        sys.stdout.write(f"  [{i+1}/{len(payloads)}] ")
        sys.stdout.flush()
        r = send_request(url, payload)
        results.append(r)
        if r["success"]:
            parts = [f"OK  {r['latency_s']:.2f}s  {r.get('text_tokens', 0)} txt_tok"]
            if r.get("ttft_s") is not None:
                parts.append(f"TTFT={r['ttft_s']:.2f}s")
            if r.get("ttlt_s") is not None:
                parts.append(f"TTLT={r['ttlt_s']:.2f}s")
            if r.get("ttfa_s") is not None:
                parts.append(f"TTFA={r['ttfa_s']:.2f}s")
            if r.get("text_tok_per_sec", 0) > 0:
                parts.append(f"txt_tok/s={r['text_tok_per_sec']:.1f}")
            print("  ".join(parts))
        else:
            print(f"FAIL  {r['error'][:80]}")

    successes = [r for r in results if r["success"]]
    failures = len(results) - len(successes)

    if not successes:
        return {"name": name, "total": len(results), "failures": failures, "error": "all requests failed"}

    latencies = [r["latency_s"] for r in successes]
    ttft_values = [r["ttft_s"] for r in successes if r.get("ttft_s") is not None]
    ttlt_values = [r["ttlt_s"] for r in successes if r.get("ttlt_s") is not None]
    ttfa_values = [r["ttfa_s"] for r in successes if r.get("ttfa_s") is not None]
    text_tps = [r["text_tok_per_sec"] for r in successes if r.get("text_tok_per_sec", 0) > 0]
    overall_tps = [r["overall_tok_per_sec"] for r in successes if r.get("overall_tok_per_sec", 0) > 0]

    def pct(vals, p):
        if len(vals) < 2:
            return vals[0] if vals else None
        return sorted(vals)[int(len(vals) * p)]

    summary = {
        "name": name,
        "total": len(results),
        "successes": len(successes),
        "failures": failures,
        "latency": {
            "mean_s": statistics.mean(latencies),
            "median_s": statistics.median(latencies),
            "p95_s": pct(latencies, 0.95),
        },
        "text_tok_per_sec": {
            "mean": statistics.mean(text_tps) if text_tps else 0,
            "median": statistics.median(text_tps) if text_tps else 0,
        },
        "overall_tok_per_sec": {
            "mean": statistics.mean(overall_tps) if overall_tps else 0,
            "median": statistics.median(overall_tps) if overall_tps else 0,
        },
        "ttft": {
            "mean_s": statistics.mean(ttft_values) if ttft_values else None,
            "median_s": statistics.median(ttft_values) if ttft_values else None,
            "p95_s": pct(ttft_values, 0.95) if ttft_values else None,
        },
        "ttlt": {
            "mean_s": statistics.mean(ttlt_values) if ttlt_values else None,
            "median_s": statistics.median(ttlt_values) if ttlt_values else None,
            "p95_s": pct(ttlt_values, 0.95) if ttlt_values else None,
        },
        "ttfa": {
            "mean_s": statistics.mean(ttfa_values) if ttfa_values else None,
            "median_s": statistics.median(ttfa_values) if ttfa_values else None,
            "p95_s": pct(ttfa_values, 0.95) if ttfa_values else None,
        },
        "prompt_tokens_mean": statistics.mean([r["prompt_tokens"] for r in successes]),
        "completion_tokens_mean": statistics.mean([r["completion_tokens"] for r in successes]),
        "text_tokens_mean": statistics.mean([r["text_tokens"] for r in successes]),
        "requests": results,
    }

    print(f"\n  Latency:      mean={summary['latency']['mean_s']:.2f}s  "
          f"median={summary['latency']['median_s']:.2f}s  "
          f"p95={summary['latency']['p95_s']:.2f}s")
    if ttft_values:
        print(f"  TTFT:         mean={summary['ttft']['mean_s']:.2f}s  "
              f"median={summary['ttft']['median_s']:.2f}s")
    if ttlt_values:
        print(f"  TTLT:         mean={summary['ttlt']['mean_s']:.2f}s  "
              f"median={summary['ttlt']['median_s']:.2f}s")
    if ttfa_values:
        print(f"  TTFA:         mean={summary['ttfa']['mean_s']:.2f}s  "
              f"median={summary['ttfa']['median_s']:.2f}s")
    if text_tps:
        print(f"  Text tok/s:   mean={summary['text_tok_per_sec']['mean']:.1f}  "
              f"median={summary['text_tok_per_sec']['median']:.1f}")
    if overall_tps:
        print(f"  Overall tok/s: mean={summary['overall_tok_per_sec']['mean']:.1f}  "
              f"median={summary['overall_tok_per_sec']['median']:.1f}")
    print(f"  Failures:     {failures}/{len(results)}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-Omni-Instruct multimodal inference")
    parser.add_argument("--url", required=True, help="Base URL of the vLLM-Omni server")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per modality")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup requests per modality (not recorded)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--output", default="benchmark_instruct.json", help="Output JSON file")
    parser.add_argument("--modalities", default="all",
                        help="Comma-delimited list of modalities to benchmark: "
                             "text,image,audio,video,tts (default: all)")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model
    url = args.url.rstrip("/")
    n = args.samples

    valid_modalities = {"text", "image", "audio", "video", "tts"}
    if args.modalities.strip().lower() == "all":
        enabled = valid_modalities
    else:
        enabled = {m.strip().lower() for m in args.modalities.split(",")}
        unknown = enabled - valid_modalities
        if unknown:
            parser.error(f"unknown modalities: {', '.join(sorted(unknown))}. "
                         f"Valid: {', '.join(sorted(valid_modalities))}")

    try:
        r = requests.get(f"{url}/health", timeout=10)
        r.raise_for_status()
        print(f"Server healthy: {url}")
    except Exception as e:
        print(f"Server not reachable at {url}: {e}")
        sys.exit(1)

    text_samples = load_text_samples(n)
    image_paths = load_image_paths(n)
    audio_paths = load_audio_paths(n)
    video_paths = load_video_paths(n)

    print(f"Loaded: {len(text_samples)} text, {len(image_paths)} image, "
          f"{len(audio_paths)} audio, {len(video_paths)} video samples")

    all_results = {}

    if "text" in enabled:
        payloads = [build_text_to_text_audio(p) for p in text_samples]
        all_results["text_to_text_audio"] = run_benchmark("Text -> Text+Audio", url, payloads, warmup=args.warmup)

    if "image" in enabled:
        payloads = [build_image_to_text_audio("Describe this image in one sentence.", p)
                    for p in image_paths]
        all_results["image_to_text_audio"] = run_benchmark("Image -> Text+Audio", url, payloads, warmup=args.warmup)

    if "audio" in enabled:
        payloads = [build_audio_to_text_audio("Transcribe this audio.", p)
                    for p in audio_paths]
        all_results["audio_to_text_audio"] = run_benchmark("Audio -> Text+Audio", url, payloads, warmup=args.warmup)

    if "video" in enabled:
        payloads = [build_video_to_text_audio("Describe what happens in this video.", p)
                    for p in video_paths]
        all_results["video_to_text_audio"] = run_benchmark("Video -> Text+Audio", url, payloads, warmup=args.warmup)

    if "tts" in enabled:
        payloads = [build_tts(p) for p in text_samples]
        all_results["tts"] = run_benchmark("Text -> Audio (TTS)", url, payloads, warmup=args.warmup)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"{'Modality':<22} {'E2E(s)':>7} {'TTFT':>7} {'TTLT':>7} {'TTFA':>7} "
          f"{'TxtTk/s':>8} {'Tok/s':>6} {'Fail':>5}")
    print(f"{'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*5}")
    for key in ["text_to_text_audio", "image_to_text_audio", "audio_to_text_audio",
                "video_to_text_audio", "tts"]:
        if key not in all_results:
            continue
        r = all_results[key]
        if "error" in r:
            print(f"{r['name']:<22} {'FAILED':>7}")
            continue
        lat = r["latency"]
        ttft_str = fmt_metric(r["ttft"]["median_s"]) if r["ttft"]["median_s"] is not None else "n/a"
        ttlt_str = fmt_metric(r["ttlt"]["median_s"]) if r["ttlt"]["median_s"] is not None else "n/a"
        ttfa_str = fmt_metric(r["ttfa"]["median_s"]) if r["ttfa"]["median_s"] is not None else "n/a"
        txt_tps = r["text_tok_per_sec"]
        ovr_tps = r["overall_tok_per_sec"]
        print(f"{r['name']:<22} {lat['median_s']:>7.2f} {ttft_str:>7} {ttlt_str:>7} "
              f"{ttfa_str:>7} {txt_tps['mean']:>8.1f} {ovr_tps['mean']:>6.1f} {r['failures']:>5}")

    # Save results
    output_path = Path(__file__).parent / args.output
    summary_results = {}
    for k, v in all_results.items():
        summary_results[k] = {kk: vv for kk, vv in v.items() if kk != "requests"}

    with open(output_path, "w") as f:
        json.dump({"summary": summary_results, "config": {"url": url, "samples": n, "model": MODEL}}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    full_path = output_path.with_suffix(".full.json")
    with open(full_path, "w") as f:
        json.dump({"results": all_results, "config": {"url": url, "samples": n, "model": MODEL}}, f, indent=2)
    print(f"Full results saved to {full_path}")


if __name__ == "__main__":
    main()
