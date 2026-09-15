#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 TRAIN_INPUT_DIR VALIDATION_INPUT_DIR OUTPUT_DIR [UMA_TASK] [REGRESSION_TASK]" >&2
  exit 2
fi

train_dir=$1
validation_dir=$2
output_dir=$3
uma_task=${4:-omat}
regression_task=${5:-e}

python -m fairchem.core.scripts.create_uma_finetune_dataset \
  --train-dir "$train_dir" --val-dir "$validation_dir" \
  --output-dir "$output_dir" --uma-task "$uma_task" \
  --regression-task "$regression_task"
