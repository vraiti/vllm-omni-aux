#!/usr/bin/env python3
"""Snapshot an instance's root and cache EBS volumes and republish them as
a new version of a named AMI.

Invoked by aws-manage's `snapshot` subcommand, which has already resolved
the alias to an instance id. Not meant to be run standalone against an
arbitrary instance without checking what AMI it's about to replace.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def aws(*args: str) -> str:
    result = subprocess.run(
        ["aws", *args, "--output", "json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"aws {' '.join(args)} failed")
    return result.stdout


def aws_json(*args: str) -> dict:
    out = aws(*args)
    if not out.strip():
        raise RuntimeError(f"aws {' '.join(args)} produced no output")
    return json.loads(out)


def find_volume(instance: dict, device_name: str) -> str:
    for mapping in instance["BlockDeviceMappings"]:
        if mapping["DeviceName"] == device_name:
            return mapping["Ebs"]["VolumeId"]
    raise RuntimeError(f"no volume attached at {device_name}")


def create_snapshot(volume_id: str, name: str, description: str) -> str:
    print(f"Snapshotting {volume_id} ({name})...")
    out = aws_json(
        "ec2", "create-snapshot",
        "--volume-id", volume_id,
        "--description", description,
        "--tag-specifications",
        f"ResourceType=snapshot,Tags=[{{Key=Name,Value={name}}}]",
    )
    return out["SnapshotId"]


def ebs_mapping_from_image(image: dict, device_name: str) -> dict:
    for mapping in image["BlockDeviceMappings"]:
        if mapping["DeviceName"] == device_name:
            return mapping
    raise RuntimeError(f"AMI has no block device mapping for {device_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--ami-name", default="vraiti-rhel10-cuda")
    parser.add_argument("--cache-device", default="/dev/sdf")
    args = parser.parse_args()

    instance = aws_json(
        "ec2", "describe-instances", "--instance-ids", args.instance_id,
    )["Reservations"][0]["Instances"][0]

    root_device_name = instance["RootDeviceName"]
    root_volume_id = find_volume(instance, root_device_name)
    cache_volume_id = find_volume(instance, args.cache_device)

    print(f"Instance:     {args.instance_id}")
    print(f"Root volume:  {root_volume_id} ({root_device_name})")
    print(f"Cache volume: {cache_volume_id} ({args.cache_device})")

    old_images = aws_json(
        "ec2", "describe-images", "--owners", "self",
        "--filters", f"Name=tag:Name,Values={args.ami_name}",
    )["Images"]
    if not old_images:
        print(f"ERROR: AMI '{args.ami_name}' not found", file=sys.stderr)
        return 1
    old_image = old_images[0]
    old_ami_id = old_image["ImageId"]
    print(f"Current AMI:  {old_ami_id}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    root_snapshot_id = create_snapshot(
        root_volume_id, f"{args.ami_name}-root-{ts}",
        f"{args.ami_name} root snapshot from {args.alias} ({ts})",
    )
    cache_snapshot_id = create_snapshot(
        cache_volume_id, f"{args.ami_name}-cache-{ts}",
        f"{args.ami_name} cache snapshot from {args.alias} ({ts})",
    )

    print("Waiting for snapshots to complete (this can take a while)...")
    aws("ec2", "wait", "snapshot-completed",
        "--snapshot-ids", root_snapshot_id, cache_snapshot_id)

    old_root_mapping = ebs_mapping_from_image(old_image, root_device_name)
    old_cache_mapping = ebs_mapping_from_image(old_image, args.cache_device)

    def replace_snapshot(mapping: dict, snapshot_id: str) -> dict:
        ebs = dict(mapping["Ebs"])
        ebs["SnapshotId"] = snapshot_id
        ebs.pop("Encrypted", None)
        return {"DeviceName": mapping["DeviceName"], "Ebs": ebs}

    new_block_device_mappings = [
        replace_snapshot(old_root_mapping, root_snapshot_id),
        replace_snapshot(old_cache_mapping, cache_snapshot_id),
    ]

    register_args = [
        "ec2", "register-image",
        "--name", old_image["Name"],
        "--architecture", old_image["Architecture"],
        "--root-device-name", root_device_name,
        "--virtualization-type", old_image["VirtualizationType"],
        "--block-device-mappings", json.dumps(new_block_device_mappings),
    ]
    if old_image.get("EnaSupport"):
        register_args += ["--ena-support"]
    if old_image.get("BootMode"):
        register_args += ["--boot-mode", old_image["BootMode"]]

    print(f"Deregistering old AMI {old_ami_id}...")
    aws("ec2", "deregister-image", "--image-id", old_ami_id)

    print("Registering new AMI...")
    new_ami_id = aws_json(*register_args)["ImageId"]

    aws("ec2", "create-tags", "--resources", new_ami_id,
        "--tags", f"Key=Name,Value={args.ami_name}")

    print()
    print(f"New AMI:      {new_ami_id}")
    print(f"Root snapshot:  {root_snapshot_id}")
    print(f"Cache snapshot: {cache_snapshot_id}")
    print(f"Old AMI {old_ami_id} deregistered; its snapshots were left in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
