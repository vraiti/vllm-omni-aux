#!/usr/bin/env python3
"""Loads vllm-omni/vllm_omni/deploy/qwen3_omni_moe_1gpu.yaml with PyYAML and
prints the parsed dict -- a smoke test that this profile's venv actually
has pyyaml installed and importable, not just that files synced correctly.
"""
import pathlib

import yaml

path = pathlib.Path("vllm-omni/vllm_omni/deploy/qwen3_omni_moe_1gpu.yaml")
print(yaml.safe_load(path.read_text()))
