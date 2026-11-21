#!/usr/bin/env bash
# Two-update reconstruction-pretraining smoke check on one GPU.
# Verifies data loading, the alignment head, DDP, and checkpoint saving.
set -euo pipefail
cd "$(dirname "$0")/.."
NPROC=1 bash scripts/train_pretrain.sh \
  configs/pretrain/reprime_b_160.yaml outputs/smoke/pretrain --smoke-steps 2
