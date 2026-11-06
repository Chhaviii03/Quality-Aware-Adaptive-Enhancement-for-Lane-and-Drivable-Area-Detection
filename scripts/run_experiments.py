"""
scripts/run_experiments.py
--------------------------
Cross-platform reproduction runner for DIP lane and drivable-area experiments.

Runs the evaluation pipeline across models and enhancement strategies without
triggering GPU training, saving all outputs into a separate reproduction directory.

Supports:
  --split     : Split to evaluate on ('train', 'val', 'test', default: 'val')
  --limit     : Restrict to N frames (useful for smoke tests and CPU debugging)
  --dry-run   : Check prerequisites and print execution plan without running inference
  --out_dir   : Output directory (default: 'results/reproduction')
  --models    : Space-separated models to evaluate (default: ['hough', 'unet'])
  --enhancements : Space-separated enhancements (default: ['none', 'adaptive'])

Usage
-----
    # Dry-run check
    python -m scripts.run_experiments --split val --dry-run

    # Quick 10-frame reproduction smoke test on CPU
    python -m scripts.run_experiments --split val --limit 10 --out_dir results/reproduction_test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def check_prerequisites(split: str, models: list[str], checkpoint_path: str = "results/checkpoints/unet_drivable_best.pt"):
    """Verify data manifests, sample images, and model checkpoints exist."""
    errors = []

    manifest_p = Path(f"data/manifests/{split}.csv")
    if not manifest_p.exists():
        errors.append(f"Data manifest missing: '{manifest_p}'. Please build or download manifests.")
    else:
        df = pd.read_csv(manifest_p)
        if df.empty:
            errors.append(f"Data manifest '{manifest_p}' is empty.")
        else:
            first_row = df.iloc[0]
            if not Path(first_row["img_path"]).exists():
                errors.append(f"Image file missing: '{first_row['img_path']}'. Check data_root / processed files.")
            if "mask_path" in first_row and pd.notna(first_row["mask_path"]) and not Path(first_row["mask_path"]).exists():
                errors.append(f"Mask file missing: '{first_row['mask_path']}'.")

    if "unet" in models:
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            errors.append(
                f"U-Net checkpoint missing: '{ckpt_p}'. "
                "Please train on Google Colab and place 'unet_drivable_best.pt' in results/checkpoints/."
            )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Cross-platform DIP reproduction runner.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames evaluated")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and verify prerequisites without running")
    parser.add_argument("--out_dir", default="results/reproduction", help="Directory for reproduction outputs")
    parser.add_argument("--models", nargs="+", default=["hough", "unet"], help="Models to evaluate")
    parser.add_argument("--enhancements", nargs="+", default=["none", "adaptive"], help="Enhancements to evaluate")
    parser.add_argument("--checkpoint", default="results/checkpoints/unet_drivable_best.pt", help="U-Net checkpoint path")

    args = parser.parse_args()

    print("=" * 65)
    print("DIP EXPERIMENT REPRODUCTION RUNNER")
    print("=" * 65)
    print(f"Target Split:       {args.split}")
    print(f"Frame Limit:        {args.limit if args.limit else 'ALL'}")
    print(f"Models:             {args.models}")
    print(f"Enhancements:       {args.enhancements}")
    print(f"Output Directory:   {args.out_dir}")
    print(f"Dry Run:            {args.dry_run}")
    print("-" * 65)

    # 1. Prerequisite verification
    errors = check_prerequisites(args.split, args.models, checkpoint_path=args.checkpoint)
    if errors:
        print("PREREQUISITE VERIFICATION FAILED:")
        for err in errors:
            print(f"  [ERROR] {err}")
        sys.exit(1)

    print("Prerequisites verified: manifests, image paths, and checkpoints exist.")

    # 2. Plan experiments
    experiment_matrix = []
    for model in args.models:
        for enh in args.enhancements:
            experiment_matrix.append((model, enh))

    print(f"Total experiments planned: {len(experiment_matrix)}")
    for idx, (m, e) in enumerate(experiment_matrix, 1):
        out_csv = Path(args.out_dir) / f"{m}_{e}_{args.split}.csv"
        print(f"  {idx}. Model: {m:<8} | Enhancement: {e:<10} -> {out_csv}")

    if args.dry_run:
        print("\n[DRY RUN] All prerequisites satisfied. Exiting without execution.")
        sys.exit(0)

    # 3. Execution
    out_dir_path = Path(args.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    from src.eval.run_eval import run_eval

    completed = []
    t_total_start = time.time()

    for idx, (model, enh) in enumerate(experiment_matrix, 1):
        out_csv = str(out_dir_path / f"{model}_{enh}_{args.split}.csv")
        print(f"\n[{idx}/{len(experiment_matrix)}] Executing {model} + {enh} on {args.split}...")

        eval_args = argparse.Namespace(
            model=model,
            config=f"configs/{model}.yaml" if Path(f"configs/{model}.yaml").exists() else None,
            enhance_cfg="configs/enhance.yaml",
            router_cfg="configs/router.yaml",
            data_cfg="configs/data.yaml",
            enhancement=enh,
            split=args.split,
            limit=args.limit,
            debug=False,
            save_stages=False,
            stage_dir="results/figures/stages",
            birds_eye=False,
            resume=args.checkpoint if model == "unet" else None,
            out_csv=out_csv,
        )

        t0 = time.time()
        try:
            res_df, sum_df = run_eval(eval_args)
            t1 = time.time()
            completed.append((model, enh, out_csv, len(res_df), t1 - t0))
        except Exception as e:
            print(f"[FAIL] Error running {model} + {enh}: {e}")
            sys.exit(1)

    total_time = time.time() - t_total_start
    print("\n" + "=" * 65)
    print("REPRODUCTION RUN COMPLETE")
    print("=" * 65)
    print(f"Total time elapsed: {total_time:.2f} s")
    print(f"{'Model':<10} {'Enhancement':<12} {'Frames':<8} {'Time (s)':<10} {'Output CSV'}")
    print("-" * 65)
    for m, e, csv_p, n_f, dur in completed:
        print(f"{m:<10} {e:<12} {n_f:<8} {dur:<10.2f} {csv_p}")
    print("=" * 65)


if __name__ == "__main__":
    main()

