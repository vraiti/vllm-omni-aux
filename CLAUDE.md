# Utility Scripts

## run-remote.sh
Resolves an SSH alias (from `~/.ssh/config.d/`), syncs all repos listed in `repos.txt` to the remote via `sync.sh`, locates the requested command in `vllm-omni-aux/`, and executes it on the remote over SSH with `HF_TOKEN` forwarded.

## sync.sh
Iterates over `repos.txt`, auto-commits and pushes each local repo, then SSHes to the remote to fetch and hard-reset to the pushed branch. Uses `ssh -n` to avoid consuming stdin from the file loop. Site-package repos fetch only the tracked remote.

## deploy.py
Kills existing GPU processes, resolves a model key (e.g. `minicpm-o`) to a HuggingFace model ID and a deploy config YAML, then launches `vllm serve --omni` with `--enforce-eager`. Polls `/health` every 2 seconds and exits 0 when the server is ready, or 1 if the process dies.

## aws-manage.sh
Manages AWS EC2 GPU instances: create, start, stop, delete, rename alias, list. Maintains SSH configs in `~/.ssh/config.d/`, updates public IPs on start, and mounts remote `/tmp/logs` via SSHFS.

## livekit_replay_client.py
Replays a captured LiveKit WebSocket session (JSONL) against a vLLM-Omni `/v1/realtime` endpoint. Sends client messages at original timing (adjustable via `--speed`), logs server responses, and optionally uploads a reference voice. Collects PCM16 audio from both client input and server output, then merges them into a stereo WAV (left=client, right=server).

## decode_trace.py
Parses token IDs from vLLM server logs (`DEBUG forward input_ids`) or MiniCPM-o-Demo logs (`[PrefillTrace]`/`[TokenTrace]`), decodes them via the HuggingFace tokenizer, and prints annotated step-by-step output showing prefill chunks and generated tokens.
