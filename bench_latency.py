#!/usr/bin/env python3
"""Send N requests to the local vLLM-Omni server and record latencies."""

import json
import time
import urllib.request

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
URL = "http://localhost:8000/v1/chat/completions"
N = 10

PROMPTS = [
    "What's 2+2?",
    "Explain gravity in one sentence.",
    "Name three primary colors.",
    "What is the capital of France?",
    "How many days are in a leap year?",
    "What does CPU stand for?",
    "Translate 'hello' to Spanish.",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "What is the square root of 144?",
]

results = []
for i in range(N):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[i]}],
        "max_tokens": 256,
    }).encode()
    req = urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    elapsed = time.monotonic() - t0

    text = body["choices"][0]["message"]["content"]
    has_audio = len(body["choices"]) > 1
    results.append({
        "request": i + 1,
        "prompt": PROMPTS[i],
        "response_text": text,
        "has_audio": has_audio,
        "latency_s": round(elapsed, 4),
    })
    print(f"[{i+1}/{N}] {elapsed:.3f}s  {text[:60]}")

with open("qwen_latencies.json", "w") as f:
    json.dump(results, f, indent=2)

avg = sum(r["latency_s"] for r in results) / len(results)
print(f"\nWrote qwen_latencies.json  avg={avg:.3f}s")
