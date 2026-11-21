#!/usr/bin/env bash
# Launch distributional post-training (OT) with torchrun.
#
# Usage:
#   INIT=/path/to/pretrain/checkpoints/latest.pth NPROC=8 \
#     bash scripts/train_posttrain.sh CONFIG OUTPUT [extra args...]
#
# If INIT is unset, the recipe's default released checkpoint is used (B-160).
set -euo pipefail
CONFIG=${1:?usage: train_posttrain.sh CONFIG OUTPUT [extra args...]}
OUTPUT=${2:?usage: train_posttrain.sh CONFIG OUTPUT [extra args...]}
shift 2
NPROC=${NPROC:-8}
ARGS=(--config "${CONFIG}" --output "${OUTPUT}")
if [[ -n "${INIT:-}" ]]; then
  ARGS+=(--init-checkpoint "${INIT}")
fi
exec torchrun --standalone --nproc_per_node="${NPROC}" posttrain.py "${ARGS[@]}" "$@"
