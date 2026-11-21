#!/usr/bin/env bash
# Launch reconstruction pretraining (Priming or RePrime) with torchrun.
#
# Usage:
#   NPROC=8 bash scripts/train_pretrain.sh CONFIG OUTPUT [extra args...]
#
# Examples:
#   NPROC=8 bash scripts/train_pretrain.sh configs/pretrain/reprime_b_160.yaml outputs/pretrain/reprime_b
#   NPROC=1 bash scripts/train_pretrain.sh configs/pretrain/priming_b_160.yaml outputs/smoke --smoke-steps 2
set -euo pipefail
CONFIG=${1:?usage: train_pretrain.sh CONFIG OUTPUT [extra args...]}
OUTPUT=${2:?usage: train_pretrain.sh CONFIG OUTPUT [extra args...]}
shift 2
NPROC=${NPROC:-8}
exec torchrun --standalone --nproc_per_node="${NPROC}" pretrain.py \
  --config "${CONFIG}" --output "${OUTPUT}" "$@"
