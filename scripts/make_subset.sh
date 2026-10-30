#!/usr/bin/env bash
# scripts/make_subset.sh
# Build BDD100K manifests and verify the subset.
#
# Usage:
#   bash scripts/make_subset.sh /path/to/bdd100k [config]
#
# Positional args:
#   $1  BDD100K root directory (must contain images/100k/ and labels/)
#   $2  (optional) config file, default configs/data.yaml
#
# Outputs:  data/manifests/{train,val,test}.csv

set -euo pipefail

BDD_ROOT="${1:?Usage: make_subset.sh <bdd100k_root> [config]}"
CONFIG="${2:-configs/data.yaml}"

echo "=== Building BDD100K subset manifests ==="
echo "  BDD root : $BDD_ROOT"
echo "  Config   : $CONFIG"

python src/data/bdd_subset.py \
    --ann_json  "$BDD_ROOT/labels/det_20/det_train.json" \
                "$BDD_ROOT/labels/det_20/det_val.json" \
    --img_root  "$BDD_ROOT/images/100k/train" \
                "$BDD_ROOT/images/100k/val" \
    --out_dir   data/manifests \
    --config    "$CONFIG"

echo ""
echo "=== Manifest summary ==="
for split in train val test; do
    n=$(tail -n +2 "data/manifests/${split}.csv" | wc -l)
    echo "  ${split}: ${n} frames"
done

echo ""
echo "Done. Commit data/manifests/ only — never commit image files."
