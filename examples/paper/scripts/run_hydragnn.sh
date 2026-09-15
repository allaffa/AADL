#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 HYDRAGNN_CONFIG.json" >&2
  exit 2
fi

export HYDRAGNN_NUM_WORKERS=${HYDRAGNN_NUM_WORKERS:-0}
python -c 'import hydragnn, sys; hydragnn.run_training(sys.argv[1])' "$1"
