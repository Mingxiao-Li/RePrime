#!/usr/bin/env bash
# Two-update OT post-training smoke check. The official recipe is sized for
# eight H200s; this script uses two ranks and activation checkpointing for a
# pipeline check only, not for producing paper numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
export PRIMING_CKPT_GEN=1
export DRIFT_COMPILE=0
INIT=${INIT:-outputs/smoke/pretrain/checkpoints/latest.pth}
NPROC=${NPROC:-2} INIT="${INIT}" bash scripts/train_posttrain.sh \
  configs/posttrain/smoke_nocompile.yaml outputs/smoke/posttrain --smoke-steps 2
