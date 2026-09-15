#!/bin/zsh
set -euo pipefail

script_directory="${0:A:h}"
repository_root="${script_directory:h:h:h}"
dataset_directory="${repository_root}/data/260911"

mkdir -p "${dataset_directory}/CreateMLCheckpoint"

exec xcrun createML objecttracker \
  --upright \
  --source "${dataset_directory}/260108-3-training.usdz" \
  --output "${dataset_directory}/260911.referenceobject" \
  --checkpoint "${dataset_directory}/CreateMLCheckpoint" \
  --csv-progress \
  --summary
