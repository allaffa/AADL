#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG [AADL experiment arguments...]" >&2
  exit 2
fi

config=$1
shift
python -m AADL.experiments "$config" "$@"
