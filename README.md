# Quality-Aware Adaptive Enhancement for Lane and Drivable-Area Detection

> **DIP Course Project · October 2026 · @Chhavi**

---

## Overview

A complete image-processing and segmentation pipeline that:

1. **Scores** each driving frame on five quality axes (brightness, contrast, sharpness, noise, haze).  
2. **Routes** it to the best enhancement (CLAHE, gamma, dehaze, denoise, denoise+CLAHE, or none).  
3. **Detects** lanes and drivable area with a classical Hough pipeline **and** two learned models (U-Net from scratch, SegFormer-B0).  
4. **Evaluates** per condition (day-clear, night, rain, synthetic fog) with bootstrap confidence intervals.

---

## Repository Layout

```
├── configs/               YAML configs for every module
│   ├── data.yaml
│   ├── enhance.yaml
│   ├── router.yaml
│   ├── unet.yaml
│   └── segformer.yaml
├── data/
│   └── manifests/         CSV manifests ONLY (no images committed)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── data/              bdd_subset.py  labels.py  degrade.py
│   ├── quality/           metrics.py
│   ├── enhance/           clahe.py  gamma.py  dehaze.py  denoise.py
│   ├── router/            rule.py  oracle.py  learned.py
│   ├── classical/         lane_hough.py
│   ├── models/            unet.py  segformer.py  train.py
│   ├── eval/              metrics.py  run_eval.py  bootstrap.py
│   └── viz/               panels.py
├── scripts/
│   ├── make_subset.sh
│   └── run_all.sh
├── results/
│   ├── tables/
│   └── figures/
├── tests/
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1 — clone and install
git clone <repo-url>
cd lane-qa-enhance
pip install -r requirements.txt

# 2 — build manifests (needs BDD100K root)
bash scripts/make_subset.sh /path/to/bdd100k

# 3 — debug run (CPU, 20 images, 2 epochs)
python src/eval/run_eval.py \
    --config configs/unet.yaml \
    --model unet \
    --enhancement adaptive \
    --split test \
    --debug

# 4 — full evaluation (GPU recommended)
bash scripts/run_all.sh
```

---

## Data

* **Source**: [BDD100K](https://bdd-data.berkeley.edu/) — download after registering; check licence for research use.  
* **Subset**: 6 000 frames, stratified by `weather` + `timeofday` tags.  
  * Train 4 200 | Val 900 | Test 900  
  * Conditions: day-clear ⅓, night ⅓, rain ⅓ (real); fog is **synthetic** (atmospheric scattering, §5.4 of PRD).  
* **No images are committed**. Only `data/manifests/*.csv` (image IDs + tags) are versioned.

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.10 + |
| CV | OpenCV 4.8 + contrib |
| DL | PyTorch 2.1, HuggingFace Transformers |
| Target compute | CPU locally; free-tier GPU (Colab / Kaggle) for training |
| Configs | All paths in YAML → point `data_root` / `checkpoint_dir` at Google Drive on Colab |

---

## Experiments

| ID | Detector | Preprocessing | Purpose |
|----|----------|--------------|---------|
| B1/B2/B3 | Hough / U-Net / SegFormer | none | Baselines |
| A1/A2/A3 | Hough / U-Net / SegFormer | CLAHE (all frames) | Fixed-enhance comparison |
| P1/P2/P3 | Hough / U-Net / SegFormer | Adaptive router | **Proposed** |
| O | U-Net | Oracle action | Router upper bound |
| E1 | all | Each enhancement fixed, per condition | Ablation |
| E2 | U-Net | Router v1 vs v2 | Router design |
| E3 | U-Net | Aug-trained, no enhance vs adaptive | Does aug replace enhance? |

---

## Metrics

* **Drivable area**: IoU, mIoU, Dice  
* **Lane F1**: pixel precision/recall within **5 px tolerance** (distance transform); our own definition — documented in report  
* **Enhancement**: PSNR / SSIM vs clear original (synthetic sets only)  
* **Router**: accuracy vs oracle; regret (oracle score − router score)  
* **Cost**: ms/frame per stage; peak GPU memory  
* **Statistics**: mean ± 95 % bootstrap CI (1 000 resamples) per condition

---

## Reproducibility

* Fixed seeds everywhere (`SEED = 42`).  
* Every training/inference script reads a YAML config.  
* All results come from `run_eval.py` → per-frame CSV → aggregation scripts.  
* **Never tune on the test split.**  
* Log every run with config snapshot + git commit hash + hardware string.

---

## Authors

| Owner | Modules |
|-------|---------|
| A | Quality analyzer, enhancement library, router, degradation, visualization |
| B | Data pipeline, label masks, classical detector, U-Net, SegFormer, evaluation harness |

---

## Licence

Code: MIT. Dataset: BDD100K licence — see [dataset page](https://bdd-data.berkeley.edu/).
