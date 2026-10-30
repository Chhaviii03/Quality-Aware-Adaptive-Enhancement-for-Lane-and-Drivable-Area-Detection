"""
src/data/bdd_subset.py
----------------------
Builds stratified train/val/test manifests from a BDD100K annotation JSON.

Usage
-----
    python src/data/bdd_subset.py \
        --ann_json /data/bdd100k/labels/det_20/det_train.json \
        --img_root /data/bdd100k/images/100k/train \
        --out_dir   data/manifests \
        --config    configs/data.yaml

Writes data/manifests/{train,val,test}.csv with columns:
    image_id, condition, split, img_path, lane_label_path, seg_label_path

Nothing is committed but the CSVs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import yaml


CONDITION_MAP = {
    # (timeofday, weather) → condition string
    ("daytime", "clear"):  "day_clear",
    ("night",   "clear"):  "night",
    ("night",   "partly cloudy"): "night",
    ("daytime", "rainy"):  "rain",
    ("night",   "rainy"):  "rain",
    ("daytime", "overcast"): "day_clear",   # treat overcast as day-clear
}


def _map_condition(attr: dict) -> str | None:
    tod     = attr.get("timeofday", "").lower()
    weather = attr.get("weather",   "").lower()
    return CONDITION_MAP.get((tod, weather))


def build_manifests(
    ann_jsons: list[Path],
    img_roots:  list[Path],
    out_dir:    Path,
    cfg:        dict,
    seed:       int = 42,
) -> dict[str, pd.DataFrame]:
    """Build stratified manifests and write CSVs.

    Parameters
    ----------
    ann_jsons : list[Path]
        BDD100K annotation JSONs (train + val, in order matching img_roots).
    img_roots : list[Path]
        Corresponding image root directories.
    out_dir : Path
        Where to write {train,val,test}.csv.
    cfg : dict
        Contents of configs/data.yaml.
    seed : int

    Returns
    -------
    dict mapping split name to DataFrame.
    """
    random.seed(seed)
    conditions = cfg["conditions"]  # {condition: fraction}
    n_train = cfg["n_train"]
    n_val   = cfg["n_val"]
    n_test  = cfg["n_test"]
    total   = n_train + n_val + n_test

    # Collect all frames with a recognised condition
    all_frames: dict[str, list[dict]] = {c: [] for c in conditions}

    for ann_json, img_root in zip(ann_jsons, img_roots):
        print(f"Loading {ann_json} …")
        with open(ann_json) as f:
            data = json.load(f)

        for frame in data:
            attr = frame.get("attributes", {})
            cond = _map_condition(attr)
            if cond not in all_frames:
                continue
            image_id = frame["name"]          # e.g. "0a0a0b1a-7c39d841.jpg"
            img_path = img_root / image_id
            if not img_path.exists():
                continue
            all_frames[cond].append({
                "image_id": image_id,
                "condition": cond,
                "img_path": str(img_path),
            })

    # Stratified sampling per condition
    records: list[dict] = []
    for cond, frac in conditions.items():
        n_cond = round(total * frac)
        pool = all_frames[cond]
        random.shuffle(pool)
        if len(pool) < n_cond:
            raise ValueError(
                f"Condition '{cond}' has only {len(pool)} frames, need {n_cond}. "
                "Reduce subset size or adjust condition fractions."
            )
        records.extend(pool[:n_cond])

    random.shuffle(records)

    # Split into train / val / test (no video-leakage check here; see note)
    train_recs = records[:n_train]
    val_recs   = records[n_train:n_train + n_val]
    test_recs  = records[n_train + n_val:n_train + n_val + n_test]

    splits = {"train": train_recs, "val": val_recs, "test": test_recs}

    out_dir.mkdir(parents=True, exist_ok=True)
    dfs: dict[str, pd.DataFrame] = {}
    for split, recs in splits.items():
        df = pd.DataFrame(recs)
        df["split"] = split
        df.to_csv(out_dir / f"{split}.csv", index=False)
        print(f"Wrote {len(df):>5} rows → {out_dir / split}.csv")
        dfs[split] = df

    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BDD100K subset manifests.")
    p.add_argument("--ann_json",  nargs="+", required=True,
                   help="BDD100K annotation JSON(s)")
    p.add_argument("--img_root",  nargs="+", required=True,
                   help="Image root dir(s) matching each annotation JSON")
    p.add_argument("--out_dir",   default="data/manifests")
    p.add_argument("--config",    default="configs/data.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_manifests(
        ann_jsons=[Path(p) for p in args.ann_json],
        img_roots=[Path(p) for p in args.img_root],
        out_dir=Path(args.out_dir),
        cfg=cfg,
        seed=cfg.get("seed", 42),
    )
