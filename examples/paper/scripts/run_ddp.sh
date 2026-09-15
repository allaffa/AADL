#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PROCESSES CONFIG [AADL experiment arguments...]" >&2
  exit 2
fi

processes=$1
config=$2
shift 2
torchrun --standalone --nproc-per-node="$processes" \
  -m AADL.experiments "$config" --set execution.distributed=true "$@"
