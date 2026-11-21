#!/usr/bin/env bash
# Official FID / FDr5 evaluation: 50,000 images, 3 sampling seeds, 4 ranks.
#
# Usage:
#   bash scripts/eval_paper.sh MODEL CHECKPOINT [OUTPUT_DIR]
# MODEL is the architecture name: priming_b_160 or priming_l_160.
# For a RePrime checkpoint, pass the matching architecture name.
set -euo pipefail
MODEL=${1:?usage: eval_paper.sh MODEL CHECKPOINT [OUTPUT_DIR]}
CHECKPOINT=${2:?usage: eval_paper.sh MODEL CHECKPOINT [OUTPUT_DIR]}
OUTPUT=${3:-outputs/eval/${MODEL}}
NPROC=${NPROC:-4}
exec torchrun --standalone --nproc_per_node="${NPROC}" eval.py \
  --model "${MODEL}" --checkpoint "${CHECKPOINT}" \
  --num-images "${NUM_IMAGES:-50000}" --seeds ${SEEDS:-1 2 3} \
  --batch "${BATCH:-32}" --output-dir "${OUTPUT}"
