#!/usr/bin/env python3
"""Manages run-remote.sh profiles: JSON files under ~/.local/run-remote/,
selectable via `run-remote.sh --profile <name>`."""
import argparse
import json
import sys
from pathlib import Path

VALID_SYNC_LABELS = {"default", "site-package", "push-only"}
PROFILE_DIR = Path.home() / ".local" / "run-remote"


def parse_args():
    parser = argparse.ArgumentParser(prog="profile.py")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    create = subparsers.add_parser(
        "create",
        usage="%(prog)s <profile-name> [--venv NAME] [--env VAR=value ...] "
              "[--host ALIAS] [--home PATH] [--local-home PATH] "
              "[--include NAME ...] "
              "[--sync PATH[:LABEL] ...] [--command CMD [args...]]",
        help="Create (or overwrite) a profile",
    )
    create.add_argument("profile_name")
    create.add_argument("--venv", default="")
    create.add_argument("--env", action="append", default=[], metavar="VAR=value")
    create.add_argument("--host", default="")
    create.add_argument("--home", default="")
    create.add_argument("--local-home", default="",
                         help="Project directory on this machine (defaults to CWD "
                              "if unset -- see run-remote.sh/sync-remote.sh)")
    create.add_argument("--include", action="append", default=[], metavar="NAME")
    create.add_argument("--sync", action="append", default=[], metavar="PATH[:LABEL]")
    # Consumes everything after it, same as create-profile.sh's `shift;
    # COMMAND_ARGS=("$@"); break` -- must be the last flag on the line.
    create.add_argument("--command", nargs=argparse.REMAINDER, default=[],
                         metavar="CMD")

    subparsers.add_parser("ls", help="List profile names")

    show = subparsers.add_parser("show", help="Print a profile's resolved JSON")
    show.add_argument("profile_name")

    return parser.parse_args()


def build_env(env_args):
    env = {}
    for kv in env_args:
        k, _, v = kv.partition("=")
        env[k] = v
    return env


def build_sync(sync_args):
    sync = {}
    for kv in sync_args:
        path, sep, label = kv.partition(":")
        if not sep:
            label = "default"
        if label not in VALID_SYNC_LABELS:
            print(f"ERROR: invalid --sync label '{label}' "
                  f"(must be {', '.join(sorted(VALID_SYNC_LABELS))})", file=sys.stderr)
            sys.exit(1)
        sync[path] = label
    return sync


def profile_path(name):
    return PROFILE_DIR / f"{name}.json"


def load_profile(name):
    path = profile_path(name)
    if not path.is_file():
        print(f"ERROR: profile '{name}' not found at {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def cmd_create(args):
    # venv/env/sync are only written when explicitly given -- an omitted key
    # lets an `--include`d profile's value show through the latest-wins merge
    # instead of being clobbered by an implicit default.
    own = {}
    if args.venv:
        own["venv"] = args.venv
    if args.env:
        own["env"] = build_env(args.env)
    if args.host:
        own["host"] = args.host
    if args.home:
        own["home"] = args.home
    if args.local_home:
        own["local-home"] = args.local_home
    if args.command:
        own["command"] = args.command
    if args.sync:
        own["sync"] = build_sync(args.sync)

    # `--include` merges immediately (simple top-level dict update,
    # latest-wins per key, e.g. `env` is replaced wholesale rather than deep-
    # merged) rather than being stored for run-remote.sh to resolve later --
    # each included profile is already fully resolved on disk (no `include`
    # key survives to this point), so this is a flat merge, applied in
    # listed order, with this profile's own keys merged in last so they win
    # over anything included.
    merged = {}
    for inc in args.include:
        merged.update(load_profile(inc))
    merged.update(own)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = profile_path(args.profile_name)
    path.write_text(json.dumps(merged, indent=2) + "\n")

    print(f"Wrote profile '{args.profile_name}' to {path}")


def cmd_ls(_args):
    if not PROFILE_DIR.is_dir():
        return
    for path in sorted(PROFILE_DIR.glob("*.json")):
        print(path.stem)


def cmd_show(args):
    profile = load_profile(args.profile_name)
    print(json.dumps(profile, indent=2))


def main():
    args = parse_args()
    {
        "create": cmd_create,
        "ls": cmd_ls,
        "show": cmd_show,
    }[args.subcommand](args)


if __name__ == "__main__":
    main()
