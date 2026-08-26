#!/bin/bash

rm $HOME/.cache
ln -s /opt/flashinfer-cache $HOME/.cache
mkdir -p /opt/dlami/nvme/huggingface
mkdir -p /opt/dlami/nvme/uv
