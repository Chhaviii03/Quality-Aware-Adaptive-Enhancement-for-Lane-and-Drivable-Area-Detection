#!/usr/bin/env bash
# scripts/run_all.sh
# One-command full reproduction of all experiment results.
#
# Usage (GPU machine / Colab):
#   bash scripts/run_all.sh [bdd_root] [ckpt_root]
#
# Positional args:
#   $1  BDD100K root (default: value in configs/data.yaml)
#   $2  Checkpoint root (default: results/checkpoints)
#
# The script runs experiments B, A, P (baselines, fixed-CLAHE, adaptive router)
# for all three detectors, then writes aggregated tables to results/tables/.
#
# IMPORTANT: This script is intended for GPU environments.
#            Add --debug to individual commands for CPU testing.

set -euo pipefail

BDD_ROOT="${1:-}"
CKPT_ROOT="${2:-results/checkpoints}"

# ── 0. Seed check ─────────────────────────────────────────────────────────
echo "=== Reproducing Quality-Aware Adaptive Enhancement Pipeline ==="
echo "Timestamp: $(date -u)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# ── 1. Build manifests (skip if already populated) ─────────────────────────
if [ -n "$BDD_ROOT" ]; then
    bash scripts/make_subset.sh "$BDD_ROOT"
else
    echo "WARNING: BDD_ROOT not set. Skipping manifest rebuild."
fi

# ── 2. Train U-Net ─────────────────────────────────────────────────────────
echo ""
echo "=== Training U-Net ==="
python src/models/train.py \
    --model unet \
    --config configs/unet.yaml \
    --checkpoint_dir "$CKPT_ROOT/unet"

# ── 3. Train SegFormer ──────────────────────────────────────────────────────
echo ""
echo "=== Training SegFormer ==="
python src/models/train.py \
    --model segformer \
    --config configs/segformer.yaml \
    --checkpoint_dir "$CKPT_ROOT/segformer"

# ── 4. Baselines (B1/B2/B3) — no enhancement ───────────────────────────────
echo ""
echo "=== Experiment B: Baselines (no enhancement) ==="
for MODEL in hough unet segformer; do
    python src/eval/run_eval.py \
        --model "$MODEL" \
        --enhancement none \
        --split test \
        --out_csv "results/tables/${MODEL}_none_test.csv"
done

# ── 5. Fixed CLAHE (A1/A2/A3) ──────────────────────────────────────────────
echo ""
echo "=== Experiment A: Fixed CLAHE ==="
for MODEL in hough unet segformer; do
    python src/eval/run_eval.py \
        --model "$MODEL" \
        --enhancement clahe \
        --split test \
        --out_csv "results/tables/${MODEL}_clahe_test.csv"
done

# ── 6. Adaptive router (P1/P2/P3) ──────────────────────────────────────────
echo ""
echo "=== Experiment P: Adaptive router ==="
for MODEL in hough unet segformer; do
    python src/eval/run_eval.py \
        --model "$MODEL" \
        --enhancement adaptive \
        --split test \
        --out_csv "results/tables/${MODEL}_adaptive_test.csv"
done

# ── 7. Per-enhancement ablation (E1) ───────────────────────────────────────
echo ""
echo "=== Experiment E1: Per-enhancement ablation ==="
for ENH in none clahe gamma dehaze denoise denoise_clahe; do
    python src/eval/run_eval.py \
        --model unet \
        --enhancement "$ENH" \
        --split test \
        --out_csv "results/tables/unet_${ENH}_test.csv"
done

# ── 8. Aggregate tables ──────────────────────────────────────────────────────
echo ""
echo "=== Aggregating results ==="
python - <<'EOF'
import pandas as pd
import glob
from src.eval.bootstrap import aggregate_by_condition

metric_cols = ["road_iou", "road_dice", "lane_f1"]
all_csvs = glob.glob("results/tables/*.csv")

summary_rows = []
for csv in sorted(all_csvs):
    df = pd.read_csv(csv)
    if not all(c in df.columns for c in metric_cols):
        continue
    tag = csv.replace("results/tables/", "").replace(".csv", "")
    agg = aggregate_by_condition(df, metric_cols)
    agg.insert(0, "experiment", tag)
    summary_rows.append(agg)

if summary_rows:
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv("results/tables/summary.csv", index=False)
    print(summary.to_string(index=False))
else:
    print("No result CSVs found yet.")
EOF

echo ""
echo "=== All done. Results in results/tables/ ==="
