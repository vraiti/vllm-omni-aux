#!/usr/bin/env bash
# Shared AWS constants for aws-manage.sh, launch-instance.sh, and friends.
# Not meant to be run directly -- source it.

AMI_NAME="vraiti-rhel10-cuda"
CACHE_DEVICE_NAME="/dev/sdf"
CACHE_VOLUME_SIZE=10
ROOT_VOLUME_SIZE=200
PROJECT_TAG="aws_manage_managed"
KEY_NAME="vraiti-ed25519"
SECURITY_GROUP="sg-04371316571a0bf71"
IAM_INSTANCE_PROFILE="vllm-omni-instance-profile"
RHEL10_AMI_OWNER="309956199498"
