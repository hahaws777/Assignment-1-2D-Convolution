#!/usr/bin/env bash
# Fetch published pretrained weights for the included official PyNET architecture.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p checkpoints
if [[ ! -f checkpoints/pynet_level_0.pth ]]; then
  curl --fail --location --retry 3 \
    https://download.ai-benchmark.com/s/A5wkJNztkpex8T4/download/pynet_level_0.pth \
    -o checkpoints/pynet_level_0.pth.part
  mv checkpoints/pynet_level_0.pth.part checkpoints/pynet_level_0.pth
fi
printf 'Pretrained weights available: checkpoints/pynet_level_0.pth\n'
