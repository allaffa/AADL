#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 FAIRCHEM_CONFIG.yaml [Hydra overrides...]" >&2
  exit 2
fi

config=$1
shift
fairchem -c "$config" "$@"
