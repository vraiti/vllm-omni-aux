#!/usr/bin/env python3
"""Fetch a /v1/realtime session dump (and its events JSONL, if present) from
a remote vLLM-Omni host, then extract every turn's audio (WAV) and every
assistant turn's transcript (txt).

The session dumper (FullDuplexRealtimeConnection._dump_conversation_history)
writes to /tmp/realtime_session_<session_id>.json on the remote host; the
unredacted send/recv event log (if that instrumentation is deployed) writes
to /tmp/realtime_events_<session_id>.jsonl in the same directory.

Usage:
    python3 fetch_and_extract_session.py sess_68a0f306c3c346bea53f1d5f
    python3 fetch_and_extract_session.py --latest
    python3 fetch_and_extract_session.py --latest --host vllm-omni
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

SAMPLE_RATE_HZ = 24000


def find_project_root() -> Path:
    """vllm-omni-aux is itself a symlink into a directory shared across
    projects (e.g. duplex-reimpl/vllm-omni-aux -> .tmpl/vllm-omni-aux), so
    neither Path(__file__).resolve() nor Path.cwd() can be used to find
    *this* project's own notes/data directories: both go through the
    kernel's getcwd(), which returns the symlink-resolved (shared, wrong)
    path the instant a shell `cd`s through it -- unlike Path.resolve(),
    switching to Path.absolute() alone does not help, since the underlying
    cwd is already canonicalized before Python ever sees it.

    The one thing that *does* still hold the pre-resolution path is the
    shell's own $PWD (bash tracks this logically, not via getcwd()) -- so
    anchor on that instead, truncating at the known vllm-omni-aux path
    component. Falls back to cwd if $PWD is unset or doesn't contain it
    (e.g. not invoked from within a shell that cd'd through the symlink).
    """
    pwd = os.environ.get("PWD", "")
    if "vllm-omni-aux" in pwd:
        return Path(pwd.split("vllm-omni-aux")[0].rstrip("/"))
    return Path.cwd()


PROJECT_ROOT = find_project_root()
SESSION_DUMPS_DIR = PROJECT_ROOT / "notes" / "session-dumps"
TURNS_DIR = PROJECT_ROOT / "data" / "turns"


def find_latest_session_id(host: str) -> str:
    result = subprocess.run(
        ["ssh", host, "ls -t /tmp/realtime_session_sess_*.json 2>/dev/null | head -1"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote_path = result.stdout.strip()
    if not remote_path:
        raise RuntimeError(f"No realtime_session_*.json files found on {host}:/tmp")
    # realtime_session_sess_XXXX.json -> sess_XXXX
    stem = Path(remote_path).stem
    return stem.removeprefix("realtime_session_")


def scp_if_exists(host: str, remote_path: str, local_path: Path) -> bool:
    check = subprocess.run(["ssh", host, f"test -f {remote_path}"])
    if check.returncode != 0:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", f"{host}:{remote_path}", str(local_path)], check=True)
    return True


def load_truncate_ms(events_jsonl_path: Path | None) -> dict[str, float]:
    """item_id -> audio_end_ms from conversation.item.truncate/.truncated
    events in the (unredacted) events JSONL -- see connection.py's
    pending_truncations_ms for why a given item can be truncated more than
    once before it resolves; last one wins here, matching the server."""
    truncate_ms: dict[str, float] = {}
    if events_jsonl_path is None or not events_jsonl_path.exists():
        return truncate_ms
    for line in events_jsonl_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") not in ("conversation.item.truncate", "conversation.item.truncated"):
            continue
        item_id = rec.get("item_id")
        audio_end_ms = rec.get("audio_end_ms")
        if item_id is not None and audio_end_ms is not None:
            truncate_ms[item_id] = audio_end_ms
    return truncate_ms


def extract_turns(session_json_path: Path, out_dir: Path, events_jsonl_path: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(session_json_path.read_text())
    items = data.get("items", [])
    truncate_ms = load_truncate_ms(events_jsonl_path)

    counters = {"user": 0, "assistant": 0}
    for item in items:
        role = item.get("role")
        if role not in counters:
            continue
        idx = counters[role]
        counters[role] += 1

        audio_b64 = None
        transcript = None
        for part in item.get("content", []):
            if part.get("audio") and audio_b64 is None:
                audio_b64 = part["audio"]
            text = part.get("transcript") or part.get("text")
            if text:
                transcript = text

        if audio_b64:
            pcm = base64.b64decode(audio_b64)

            # The stored item's audio is always the *full* generation, kept
            # for observability even after a conversation.item.truncate --
            # only the transcript gets trimmed server-side. Split the audio
            # itself here to match: {role}_{idx}.wav becomes just the heard
            # prefix (what the client actually played), and the unheard
            # remainder -- generated but never played -- goes to
            # {role}_{idx}_tail.wav, so a glitch check isn't accidentally
            # listening past the point playback was ever cut off.
            audio_end_ms = truncate_ms.get(item.get("id"))
            if audio_end_ms is not None:
                split_sample = int(audio_end_ms / 1000 * SAMPLE_RATE_HZ)
                split_byte = split_sample * 2
                head_pcm, tail_pcm = pcm[:split_byte], pcm[split_byte:]
            else:
                head_pcm, tail_pcm = pcm, b""

            wav_path = out_dir / f"{role}_{idx}.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE_HZ)
                wf.writeframes(head_pcm)
            print(f"wrote {wav_path} ({len(head_pcm) / 2 / SAMPLE_RATE_HZ:.2f}s)")

            if tail_pcm:
                tail_path = out_dir / f"{role}_{idx}_tail.wav"
                with wave.open(str(tail_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE_HZ)
                    wf.writeframes(tail_pcm)
                print(f"wrote {tail_path} ({len(tail_pcm) / 2 / SAMPLE_RATE_HZ:.2f}s, unheard/truncated tail)")
        else:
            print(f"skip {role}_{idx}: no audio")

        # User turns have no server-side transcript (raw audio only, never
        # whisper-transcribed here) -- only write a .txt for assistant turns.
        if role == "assistant":
            txt_path = out_dir / f"{role}_{idx}.txt"
            txt_path.write_text(transcript or "")
            print(f"wrote {txt_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_id", nargs="?", help="e.g. sess_68a0f306c3c346bea53f1d5f")
    parser.add_argument("--latest", action="store_true", help="Fetch the most recently modified session dump instead")
    parser.add_argument("--host", default="vllm-omni", help="SSH alias for the remote vLLM-Omni host (default: vllm-omni)")
    args = parser.parse_args()

    if not args.session_id and not args.latest:
        parser.error("Provide a session_id or --latest")

    session_id = args.session_id or find_latest_session_id(args.host)
    print(f"Session: {session_id}")

    session_json = SESSION_DUMPS_DIR / f"realtime_session_{session_id}.json"
    events_jsonl = SESSION_DUMPS_DIR / f"realtime_events_{session_id}.jsonl"

    if not scp_if_exists(args.host, f"/tmp/realtime_session_{session_id}.json", session_json):
        print(f"ERROR: no session dump found for {session_id} on {args.host}:/tmp", file=sys.stderr)
        return 1
    print(f"Fetched session dump: {session_json}")

    have_events = scp_if_exists(args.host, f"/tmp/realtime_events_{session_id}.jsonl", events_jsonl)
    if have_events:
        print(f"Fetched events log: {events_jsonl}")
    else:
        print("(no events JSONL found for this session -- older session, or the session ended before that instrumentation was deployed)")

    out_dir = TURNS_DIR / session_id
    extract_turns(session_json, out_dir, events_jsonl if have_events else None)
    print(f"\nDone. Turns extracted to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
