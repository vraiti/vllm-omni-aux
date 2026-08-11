#!/usr/bin/env python3
"""Decode token IDs from vLLM or MiniCPM-o-Demo server logs.

Parses 'DEBUG forward input_ids (nonzero): [...]' lines from vLLM logs
and '[PrefillTrace]'/'[TokenTrace]'/'[FinalizeTrace]' lines from demo logs,
then decodes them using the MiniCPM-o tokenizer.

Usage:
    decode_trace.py <logfile> [--model openbmb/MiniCPM-o-4_5]
"""

import argparse
import ast
import re
import sys

VLLM_FORWARD_RE = re.compile(
    r"DEBUG forward input_ids \(nonzero\): (\[[\d, ]+\])"
)
DEMO_PREFILL_RE = re.compile(
    r"\[PrefillTrace\] chunk=(\d+) mode=(\w+) token_ids=(\[[\d, ]*\])"
)
DEMO_TOKEN_RE = re.compile(
    r"\[TokenTrace\] t=(\d+) is_listen=(\w+) text_tokens=(\d+)"
)
DEMO_TOKEN_STEP_RE = re.compile(
    r"j=(\d+) (\w+) id=(\d+) '([^']*)'"
)
DEMO_FINALIZE_RE = re.compile(
    r"\[FinalizeTrace\] feed token_ids=(\[[\d, ]+\])"
)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def decode_ids(tokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.decode([tid]) for tid in ids]


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def process_vllm_log(lines: list[str], tokenizer):
    step = 0
    for line in lines:
        line = strip_ansi(line)
        m = VLLM_FORWARD_RE.search(line)
        if not m:
            continue
        ids = ast.literal_eval(m.group(1))
        decoded = decode_ids(tokenizer, ids)
        step += 1
        if len(ids) > 5:
            print(f"[step {step:3d}] prefill ({len(ids)} tokens)")
            for tid, text in zip(ids, decoded):
                print(f"  {tid:>8d}  {text!r}")
        else:
            pairs = " ".join(f"{tid}={text!r}" for tid, text in zip(ids, decoded))
            print(f"[step {step:3d}] {pairs}")


def process_demo_log(lines: list[str], tokenizer):
    for line in lines:
        line = strip_ansi(line)

        m = DEMO_PREFILL_RE.search(line)
        if m:
            chunk, mode, ids_str = m.group(1), m.group(2), m.group(3)
            ids = ast.literal_eval(ids_str)
            decoded = decode_ids(tokenizer, ids)
            pairs = " ".join(f"{tid}={text!r}" for tid, text in zip(ids, decoded))
            print(f"[chunk {chunk}] prefill mode={mode} {pairs}")
            continue

        m = DEMO_TOKEN_RE.search(line)
        if m:
            t, is_listen = m.group(1), m.group(2)
            print(f"[chunk {t}] generate is_listen={is_listen}")
            continue

        m = DEMO_TOKEN_STEP_RE.search(line)
        if m:
            j, kind, tid_str, tok_text = m.groups()
            tid = int(tid_str)
            decoded = tokenizer.decode([tid])
            print(f"  j={j} {kind} {tid}={decoded!r} (log: '{tok_text}')")
            continue

        m = DEMO_FINALIZE_RE.search(line)
        if m:
            ids = ast.literal_eval(m.group(1))
            decoded = decode_ids(tokenizer, ids)
            pairs = " ".join(f"{tid}={text!r}" for tid, text in zip(ids, decoded))
            print(f"  finalize {pairs}")
            continue


def detect_format(lines: list[str]) -> str:
    for line in lines:
        line = strip_ansi(line)
        if "DEBUG forward input_ids" in line:
            return "vllm"
        if "[PrefillTrace]" in line or "[TokenTrace]" in line:
            return "demo"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfile", help="Path to server log file")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5",
                        help="HuggingFace model ID for tokenizer")
    parser.add_argument("--format", choices=["vllm", "demo", "auto"],
                        default="auto", help="Log format (default: auto-detect)")
    args = parser.parse_args()

    with open(args.logfile) as f:
        lines = f.readlines()

    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(lines)
        if fmt == "unknown":
            print("Could not detect log format", file=sys.stderr)
            sys.exit(1)
        print(f"Detected format: {fmt}", file=sys.stderr)

    print("Loading tokenizer...", file=sys.stderr)
    tokenizer = load_tokenizer(args.model)
    print("Tokenizer loaded.", file=sys.stderr)

    if fmt == "vllm":
        process_vllm_log(lines, tokenizer)
    else:
        process_demo_log(lines, tokenizer)


if __name__ == "__main__":
    main()
