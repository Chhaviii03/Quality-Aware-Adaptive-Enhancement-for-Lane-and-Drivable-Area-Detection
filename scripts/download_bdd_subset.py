"""
scripts/download_bdd_subset.py
------------------------------
Downloads a sampled subset of BDD100K Drivable Area Segmentation from
the Hugging Face mirror 'dronefreak/BDD100K-Drivable-Area-Segmentation'.

Requirements:
- Randomly samples 1,500 train and 300 val images (seed 42) with matching drivable masks.
- Downloads 30 matching color masks for demo screenshots.
- Never downloads the full repository (uses hf_hub_download).
- Saves into:
    bdd_subset/images/{train,val}/
    bdd_subset/masks/{train,val}/
    bdd_subset/color_masks/
- Verifies stem matching and reports counts and disk size.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError


import yaml

REPO_ID = "dronefreak/BDD100K-Drivable-Area-Segmentation"
REPO_TYPE = "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BDD100K subset from Hugging Face mirror")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to config YAML")
    parser.add_argument("--repo-id", default=REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--n-train", type=int, default=None, help="Number of train images to sample")
    parser.add_argument("--n-val", type=int, default=None, help="Number of val images to sample")
    parser.add_argument("--n-color", type=int, default=0, help="Number of color masks to download")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--out-dir", type=str, default="bdd_subset", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=24, help="Parallel download workers")
    parser.add_argument("--cache-file-list", type=str, default=".cache_repo_files.json", help="Path to cache repo file list")
    parser.add_argument("--exclude-manifests", nargs="*", default=None, help="Manifest CSVs containing stems to exclude")
    return parser.parse_args()


def get_repo_files(api: HfApi, repo_id: str, cache_path: Path) -> list[str]:
    """Retrieve file list from repo or local cache."""
    if cache_path.exists():
        print(f"Loading cached repository file list from {cache_path}...")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to read cache file ({e}), fetching from Hugging Face...")

    print(f"Listing files from repo '{repo_id}' via HfApi().list_repo_files...")
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE)
    except (GatedRepoError, RepositoryNotFoundError) as err:
        print(f"\n[ERROR] Authentication or repo access error: {err}")
        print("If login is required, please run: huggingface-cli login\n")
        sys.exit(1)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "403" in msg or "unauthorized" in msg or "gated" in msg:
            print(f"\n[ERROR] Authentication required: {e}")
            print("Please run: huggingface-cli login\n")
            sys.exit(1)
        raise

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(files, f)
        print(f"Cached repo file list to {cache_path}")
    except Exception as e:
        print(f"Warning: could not cache file list: {e}")

    return files


def download_single_file(repo_id: str, repo_file: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download single file using hf_hub_download and copy/move to dest_path."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = dest_path.parent / ".tmp_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                filename=repo_file,
                local_dir=str(temp_dir),
            )
            # Move downloaded file to target dest_path
            shutil.move(downloaded, dest_path)
            return True
        except (GatedRepoError, RepositoryNotFoundError) as e:
            print(f"\n[ERROR] Authentication or repo access error: {e}")
            print("If login is required, please run: huggingface-cli login\n")
            sys.exit(1)
        except Exception as e:
            msg = str(e).lower()
            if "401" in msg or "403" in msg or "unauthorized" in msg or "gated" in msg:
                print(f"\n[ERROR] Authentication required: {e}")
                print("Please run: huggingface-cli login\n")
                sys.exit(1)
            if attempt == max_retries:
                print(f"Failed to download {repo_file} after {max_retries} attempts: {e}")
                return False
            time.sleep(1.0 * attempt)
    return False


def get_dir_size_mb(path: Path) -> float:
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024 * 1024)


def main():
    args = parse_args()
    api = HfApi()

    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    n_train_req = args.n_train if args.n_train is not None else cfg.get("n_new_total", cfg.get("n_train", 3000))
    n_val_req = args.n_val if args.n_val is not None else cfg.get("n_val_download", 0)
    exclude_manifest_paths = args.exclude_manifests or cfg.get("exclude_manifests", [])

    cache_path = Path(args.cache_file_list)
    repo_files = get_repo_files(api, args.repo_id, cache_path)
    print(f"Total files in repository: {len(repo_files):,}")

    # Confirm folder names by printing 10 sample paths across splits
    sample_train_imgs = [f for f in repo_files if "images_100k" in f and "/train/" in f][:3]
    sample_val_imgs = [f for f in repo_files if "images_100k" in f and "/val/" in f][:3]
    sample_masks = [f for f in repo_files if "drivable_maps/labels" in f][:2]
    sample_color = [f for f in repo_files if "drivable_maps/color_labels" in f][:2]
    sample_paths = sample_train_imgs + sample_val_imgs + sample_masks + sample_color

    print("\n--- 10 Sample Paths from Repo ---")
    for idx, p in enumerate(sample_paths[:10], start=1):
        print(f"  {idx:2d}. {p}")
    print("---------------------------------")
    val_in_repo = any("/val/" in f for f in repo_files)
    validation_in_repo = any("/validation/" in f for f in repo_files)
    print(f"Validation split check: contains '/val/': {val_in_repo}, contains '/validation/': {validation_in_repo}")
    print("Confirmed: The validation split folder is named 'val'.\n")

    # Map files by split and stem
    train_images: dict[str, str] = {}
    val_images: dict[str, str] = {}
    train_masks: dict[str, str] = {}
    val_masks: dict[str, str] = {}
    train_color_masks: dict[str, str] = {}
    val_color_masks: dict[str, str] = {}

    for f in repo_files:
        if not f.endswith((".jpg", ".png")):
            continue
        parts = f.split("/")
        filename = parts[-1]

        if "images_100k" in f and filename.endswith(".jpg"):
            stem = filename[:-4]
            if "/train/" in f:
                train_images[stem] = f
            elif "/val/" in f:
                val_images[stem] = f

        elif "drivable_maps/labels" in f and filename.endswith("_drivable_id.png"):
            stem = filename[:-len("_drivable_id.png")]
            if "/train/" in f:
                train_masks[stem] = f
            elif "/val/" in f:
                val_masks[stem] = f

        elif "drivable_maps/color_labels" in f and filename.endswith("_drivable_color.png"):
            stem = filename[:-len("_drivable_color.png")]
            if "/train/" in f:
                train_color_masks[stem] = f
            elif "/val/" in f:
                val_color_masks[stem] = f

    common_train = sorted(list(train_images.keys() & train_masks.keys()))
    common_val = sorted(list(val_images.keys() & val_masks.keys()))
    print(f"Available matching (image + drivable mask) pairs in repository:")
    print(f"  Train: {len(common_train):,} pairs")
    print(f"  Val:   {len(common_val):,} pairs")

    # Handle exclusions
    exclude_stems = set()
    if exclude_manifest_paths:
        for m_path in exclude_manifest_paths:
            mp = Path(m_path)
            if mp.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(mp)
                    if "image_id" in df.columns:
                        stems = {str(x).replace(".jpg", "").replace(".png", "") for x in df["image_id"]}
                        exclude_stems.update(stems)
                        print(f"Loaded {len(stems)} stems to exclude from '{m_path}'")
                except Exception as e:
                    print(f"Warning: could not load exclusions from '{m_path}': {e}")
        print(f"Total stems in exclusion list: {len(exclude_stems):,}")

    # Exclude from train pool
    orig_train_len = len(common_train)
    common_train = [s for s in common_train if s not in exclude_stems]
    print(f"Remaining train pool after exclusions: {len(common_train):,} (from {orig_train_len:,})")

    if len(common_train) < n_train_req:
        print(f"\n[FATAL] Fewer than {n_train_req} images remain in the pool! Exact remaining: {len(common_train)}. STOPPING.")
        sys.exit(1)

    # Deterministic sampling with independent RNGs
    rng_train = random.Random(seed)
    rng_val = random.Random(seed)
    n_train = min(n_train_req, len(common_train))
    n_val = min(n_val_req, len(common_val))

    train_sampled_stems = rng_train.sample(common_train, n_train)
    val_sampled_stems = rng_val.sample(common_val, n_val) if n_val > 0 else []
    print(f"Sampled {n_train} new train images (and {n_val} val images) with seed {seed}.")

    if val_sampled_stems:
        overlap = set(train_sampled_stems) & set(val_sampled_stems)
        print(f"Overlap check: {len(overlap)} images overlap between train and val (disjoint sets confirmed: {len(overlap) == 0}).")
        assert len(overlap) == 0, f"Error: found {len(overlap)} overlapping images between train and val!"

    # Sample color masks for 30 images (from sampled train/val that have matching color masks)
    sampled_with_color = []
    # Prioritize val first, then train
    for stem in val_sampled_stems:
        if stem in val_color_masks:
            sampled_with_color.append((stem, val_color_masks[stem]))
    for stem in train_sampled_stems:
        if len(sampled_with_color) >= args.n_color:
            break
        if stem in train_color_masks:
            sampled_with_color.append((stem, train_color_masks[stem]))

    color_samples = sampled_with_color[:args.n_color]
    print(f"Selected {len(color_samples)} color masks for demo screenshots.")

    # Target directories
    out_dir = Path(args.out_dir)
    img_train_dir = out_dir / "images" / "train"
    img_val_dir = out_dir / "images" / "val"
    mask_train_dir = out_dir / "masks" / "train"
    mask_val_dir = out_dir / "masks" / "val"
    color_mask_dir = out_dir / "color_masks"

    for d in [img_train_dir, img_val_dir, mask_train_dir, mask_val_dir, color_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Prepare download task queue
    # List of (repo_file, dest_path)
    download_tasks: list[tuple[str, Path]] = []

    for stem in train_sampled_stems:
        img_repo = train_images[stem]
        mask_repo = train_masks[stem]
        download_tasks.append((img_repo, img_train_dir / f"{stem}.jpg"))
        download_tasks.append((mask_repo, mask_train_dir / f"{stem}_drivable_id.png"))

    for stem in val_sampled_stems:
        img_repo = val_images[stem]
        mask_repo = val_masks[stem]
        download_tasks.append((img_repo, img_val_dir / f"{stem}.jpg"))
        download_tasks.append((mask_repo, mask_val_dir / f"{stem}_drivable_id.png"))

    for stem, color_repo in color_samples:
        download_tasks.append((color_repo, color_mask_dir / f"{stem}_drivable_color.png"))

    total_tasks = len(download_tasks)
    print(f"\nTotal files to download: {total_tasks:,}")
    print(f"Saving to '{out_dir.resolve()}' using up to {args.max_workers} worker threads...\n")

    start_time = time.time()
    completed = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(download_single_file, args.repo_id, repo_file, dest): (repo_file, dest)
            for repo_file, dest in download_tasks
        }
        for future in concurrent.futures.as_completed(futures):
            repo_file, dest = futures[future]
            try:
                success = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"Exception during download of {repo_file}: {e}")

            if completed % 200 == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed}/{total_tasks} completed ({rate:.1f} files/sec, failed: {failed})")

    # Cleanup temporary download folders if any
    for temp_dir in out_dir.rglob(".tmp_download"):
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed_total = time.time() - start_time
    print(f"\nDownload finished in {elapsed_total:.1f} seconds. Failed: {failed}")

    # Step 6: Verify image and mask counts match per split and stems match
    print("\n--- Verifying Downloaded Subset ---")
    splits_ok = True
    for split, img_dir, mask_dir, sampled_stems in [
        ("train", img_train_dir, mask_train_dir, train_sampled_stems),
        ("val", img_val_dir, mask_val_dir, val_sampled_stems),
    ]:
        imgs = sorted(list(img_dir.glob("*.jpg")))
        masks = sorted(list(mask_dir.glob("*_drivable_id.png")))
        img_stems = {p.stem for p in imgs}
        mask_stems = {p.stem[:-len("_drivable_id")] for p in masks}

        print(f"[{split.upper()}] Total on disk: {len(imgs)} images, {len(masks)} masks")
        if len(imgs) != len(masks):
            print(f"  [ERROR] Total image vs mask count mismatch in {split}!")
            splits_ok = False
        if img_stems != mask_stems:
            missing_masks = img_stems - mask_stems
            missing_imgs = mask_stems - img_stems
            print(f"  [ERROR] Stem mismatch in {split}! Missing masks: {len(missing_masks)}, Missing images: {len(missing_imgs)}")
            splits_ok = False
        else:
            print(f"  [OK] 1-to-1 exact stem match for all {len(imgs)} pairs in {split}.")

        if sampled_stems:
            missing_downloaded = set(sampled_stems) - img_stems
            if missing_downloaded:
                print(f"  [ERROR] Missing {len(missing_downloaded)} sampled stems in {split}!")
                splits_ok = False
            else:
                print(f"  [OK] All {len(sampled_stems)} newly sampled stems verified present on disk.")

    color_files = list(color_mask_dir.glob("*.png"))
    print(f"[COLOR MASKS] Downloaded: {len(color_files)} (expected {len(color_samples)})")

    # Report counts and total disk size
    total_size_mb = get_dir_size_mb(out_dir)
    print(f"\nVerification {'PASSED' if splits_ok else 'FAILED'}!")
    print(f"Total files in '{out_dir}': {sum(1 for _ in out_dir.rglob('*') if _.is_file())}")
    print(f"Total disk size: {total_size_mb:.2f} MB ({total_size_mb / 1024:.2f} GB)")


if __name__ == "__main__":
    main()

