#!/usr/bin/env bash
set -euo pipefail

# Rebuilds python-tracer's CPython (and d3g-postprocess) inside a UBI10
# container matching the RHEL 10 deploy target's glibc -- building directly
# on this machine links extension modules against whatever newer glibc it
# has, which the RHEL 10 target can't satisfy at runtime (confirmed:
# `GLIBC_2.42' not found` for termios.cpython-314-x86_64-linux-gnu.so from
# a host build). Then replaces ../d3g (the repos.txt-registered directory
# that gets synced to remotes) with the fresh build output.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_DIR="$PWD"
while [[ "$PROJECT_DIR" != "$HOME/omni" && "$PROJECT_DIR" != "/" ]]; do
    if [[ "$(dirname "$PROJECT_DIR")" == "$HOME/omni" ]]; then
        break
    fi
    PROJECT_DIR="$(dirname "$PROJECT_DIR")"
done

TRACER_DIR="$PROJECT_DIR/python-tracer"
IMAGE_TAG="python-tracer-builder"

podman build -t "$IMAGE_TAG" -f "$TRACER_DIR/Containerfile" "$TRACER_DIR"

# Force a fresh ./configure inside the container. cpython/Makefile (if left
# over from a host build, or a previous container run against a different
# mount path) hardcodes an absolute --prefix baked in at configure time; if
# that prefix isn't /src/build, `make altinstall` below installs outside the
# bind mount, into the container's throwaway overlay, and the interpreter
# never actually reaches the host's build/ directory (confirmed: prefix left
# over from a host build pointed at the host's own absolute path, which
# doesn't exist inside the container's filesystem).
rm -f "$TRACER_DIR/cpython/Makefile"

mkdir -p "$TRACER_DIR/.cargo-cache"
podman run --rm \
    -v "$TRACER_DIR:/src:Z" \
    -v "$TRACER_DIR/.cargo-cache:/root/.cargo/registry:Z" \
    -w /src \
    "$IMAGE_TAG" \
    make "${@:-build}"

echo "Replacing $PROJECT_DIR/d3g with $TRACER_DIR/build..."
rm -rf "$PROJECT_DIR/d3g"
mv "$TRACER_DIR/build" "$PROJECT_DIR/d3g"
