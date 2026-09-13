#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
./venv/bin/python -m src.train --experiment canonical \
  --epochs 20 --num-points 256 --seed 42 --device auto