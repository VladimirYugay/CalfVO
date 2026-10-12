#!/bin/bash
# Downloads and unpacks the nine TUM RGB-D freiburg1 sequences this repo tests on.
#
# Usage:  bash scripts/data/download_tum.sh /path/to/raw/tum
set -euo pipefail

TUM_PATH="${1:?pass the directory to download into, e.g. /path/to/raw/tum}"
mkdir -p "$TUM_PATH"
cd "$TUM_PATH"

BASE="https://cvg.cit.tum.de/rgbd/dataset/freiburg1"
declare -a evalset=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
    rgbd_dataset_freiburg1_rpy
)

for sequence in "${evalset[@]}"; do
    if [ -d "$sequence" ]; then
        echo "==> $sequence already unpacked, skipping"
        continue
    fi
    echo "==> $sequence"
    wget -c "$BASE/$sequence.tgz"
    tar -xzf "$sequence.tgz"
    rm -f "$sequence.tgz"
done

echo "Done. Now run:"
echo "  python scripts/data/process_tum.py $TUM_PATH /path/to/datasets/tum_videos"
