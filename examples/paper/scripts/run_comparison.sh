#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG OUTPUT_DIRECTORY [additional overrides...]" >&2
  exit 2
fi

config=$1
output=$2
shift 2

python -m AADL.experiments "$config" --output "$output/plain" \
  --set method.name=plain --set 'method.acceleration={}' "$@"
python -m AADL.experiments "$config" --output "$output/anderson-full" \
  --set method.name=anderson-full --set method.acceleration.sketch_fraction=1.0 \
  --set method.acceleration.sketch_policy=fixed "$@"
python -m AADL.experiments "$config" --output "$output/anderson-adaptive" "$@"
