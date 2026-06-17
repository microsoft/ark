#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=$PWD/python ARK_ROOT=$PWD python3 -m examples.qwen3.bench_allreduce --world-size 8
