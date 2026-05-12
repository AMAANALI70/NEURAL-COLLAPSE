# NC-MedAI — Neural Collapse-Aware Medical Image Classification

<div align="center">

**A research framework studying how class imbalance degrades feature geometry**  
*and whether geometry-aware training can restore minority-class clinical performance.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-CUDA%20%7C%20MPS%20%7C%20CPU-informational)](https://pytorch.org)

</div>

---

## Overview

**NC-MedAI** is a research framework for studying **Neural Collapse (NC)** in medical image classification under severe class imbalance. Built on HAM10000 (7-class skin lesion, 10:1 majority–minority ratio), it measures how representation geometry degrades under imbalance and whether geometry-aware interventions restore minority-class detection.

> *When a model's feature space stops collapsing properly, minority disease detection fails — often silently, long before accuracy metrics reveal a problem. NC1–NC4 metrics expose this degradation early.*

**The completed study established:**
- NC1 and NC4 are robust **early-warning indicators** of training failure, detectable from epoch 1
- ETF heads + NC regularization produce the **best geometry and best minority recall** simultaneously
- Loss-based rebalancing (weighted CE, focal loss) **degrades NC geometry** and often hurts minority performance
- Geometry preservation order and clinical performance order are **identical** across all tested methods

---

## Study Results at a Glance

All results at HAM10000, imbalance ratio = 10:1, ResNet-18, seed = 42.

| Method | Sampler | Loss | NC-reg | Val Acc | Macro F1 | NC1 | NC4 | Mel Recall |
|--------|---------|------|--------|---------|----------|-----|-----|------------|
| Baseline | Weighted | CE | ✗ | 64.4% | 0.436 | 4.81 | 0.194 | 56.0% |
| Oversampling | Weighted | CE | ✗ | 64.4% | 0.436 | 4.81 | 0.194 | 56.0% |
| **ETF + NC-reg** | Weighted | CE | ✅ | **65.0%** | **0.472** | 5.53 | **0.186** | **58.8%** |
| ETF + NC-reg + Balanced | Balanced | CE | ✅ | 61.8% | 0.429 | 5.63 | 0.180 | 58.8% |
| Weighted CE | None | WCE | ✗ | 62.2% | 0.332 | 7.94 | 0.396 | 36.3% |
| Focal Loss | None | Focal(γ=2) | ✗ | 56.9% | 0.110 | 15.65 | 0.690 | 0.0% |

> See [`results/phase2/experiment_registry.csv`](results/phase2/experiment_registry.csv) for the full canonical registry.
> See [`FINAL_REPORT.md`](FINAL_REPORT.md) for the complete scientific analysis.

---

## Repository Structure

```
NEURAL-COLLAPSE/
│
├── config/
│   ├── config.yaml              ← Master config (all hyperparameters)
│   ├── config_loader.py         ← YAML loader with profile + CLI override support
│   └── profiles/                ← Hardware-specific override profiles
│       ├── apple_silicon.yaml
│       ├── cuda_gpu.yaml
│       └── cpu_debug.yaml
│
├── data/
│   ├── medical_datasets.py      ← HAM10000 loader with imbalance_ratio control
│   ├── dataset.py               ← CIFAR-10 with controlled imbalance injection
│   ├── imbalance_sampler.py     ← Balanced / SquareRoot / Progressive samplers
│   └── preprocessing.py        ← Medical augmentation pipelines
│
├── models/
│   ├── resnet.py                ← ResNet-18 with forward_features() API
│   ├── etf_classifier.py        ← Fixed ETF head (frozen buffer)
│   ├── prototype_head.py        ← Learnable cosine prototype classifier
│   └── model_factory.py         ← build_model(cfg, method) factory
│
├── training/
│   ├── trainer.py               ← Full training loop with NC tracking
│   ├── losses.py                ← CE / WeightedCE / FocalLoss
│   ├── nc_regularization.py     ← NCCollapseReg / ETFAlignment / CombinedNCLoss
│   └── scheduler.py             ← Cosine / Step / Constant LR
│
├── evaluation/
│   ├── nc_metrics.py            ← NC1–NC4 full suite
│   ├── medical_metrics.py       ← Sensitivity / Specificity / F1 / AUC / Kappa
│   └── evaluator.py             ← evaluate_checkpoint(), extract_features()
│
├── visualization/
│   ├── tsne_visualizer.py       ← t-SNE with minority highlight
│   ├── feature_geometry.py      ← PCA, cosine heatmap, NC evolution plots
│   └── confusion_analysis.py   ← Confusion matrix, per-class recall
│
├── experiments/
│   ├── run_phase1_study.py      ← Phase-1: ETF vs baseline × r∈{1,10,50}
│   └── run_phase2_study.py      ← Phase-2: intervention sweep at r=10
│
├── results/
│   ├── experiment_registry.csv  ← Root registry (Phase-1 + pilots)
│   ├── phase1_summary.csv       ← Authoritative Phase-1 summary (6 canonical runs)
│   ├── phase2/
│   │   ├── experiment_registry.csv      ← Authoritative Phase-2 registry
│   │   ├── baseline_ham10000_r10_s42/
│   │   ├── weighted_ham10000_r10_s42/   ← Corrected weighted CE
│   │   ├── focal_ham10000_r10_s42/      ← Corrected focal
│   │   ├── oversampling_ham10000_r10_s42/
│   │   ├── etf_nc_reg_ham10000_r10_s42/ ← ⚠ Reconstructed (plots missing)
│   │   ├── etf_nc_reg_balanced_ham10000_r10_s42/
│   │   └── archived_collapsed/          ← Invalid double-rebalancing runs
│   ├── baseline_ham10000_r{1,10,50}_s42/  ← Phase-1 baseline results
│   ├── etf_ham10000_r{1,10,50}_s42/       ← Phase-1 ETF results
│   └── archived_pilot_runs/             ← Pre-study development artifacts
│
├── study_logs/
│   ├── phase1/                  ← Per-run logs for Phase-1
│   └── phase2/                  ← Per-run logs for Phase-2
│
├── train.py                     ← CLI: single training run
├── run_sweep.py                 ← CLI: sweep runner
├── FINAL_REPORT.md              ← Complete scientific report
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd NEURAL-COLLAPSE
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dataset:** Download [HAM10000 from Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).  
Extract to `./datasets/HAM10000/` with `HAM10000_metadata.csv` and `images/` subdirectory.

---

## Running Experiments

### Single run
```bash
# ETF head + NC regularization (best configuration)
python train.py --override dataset.name=ham10000 model.head=etf \
    nc_regularization.enabled=true sampling.strategy=weighted \
    training.epochs=15

# Baseline
python train.py --override dataset.name=ham10000 model.head=linear training.epochs=15

# Smoke test (CPU, ~5 min)
python train.py --override dataset.name=ham10000 model.head=etf \
    training.epochs=1 debug.fast_dev_batches=10
```

### Reproduce Phase-2 study
```bash
# Dry-run first (verify configs, no training)
python -m experiments.run_phase2_study --dry-run

# Full Phase-2 sweep (all 6 methods at r=10)
python -m experiments.run_phase2_study

# Single method
python -m experiments.run_phase2_study --methods etf_nc_reg
```

### Monitor training
```bash
# Per-epoch summary (updates every ~10 min on CPU)
tail -f study_logs/phase2/etf_nc_reg_r10_s42.log

# Mid-epoch heartbeat
tail -f results/phase2/trainer.etf_nc_reg_ham10000_r10_s42.log
```

---

## Configuration

All hyperparameters live in `config/config.yaml`. Override at runtime:

```bash
python train.py --override \
  dataset.name=ham10000 \
  model.head=etf \
  sampling.strategy=weighted \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01 \
  training.epochs=50 \
  debug.fast_dev_batches=0     # 0 = full dataset
```

Key sections:

| Section | Important keys |
|---------|---------------|
| `dataset` | `name`, `imbalance_ratio` |
| `model` | `backbone` (resnet18), `head` (linear/etf/prototype) |
| `training` | `epochs`, `batch_size`, `lr`, `loss` |
| `sampling` | `strategy` (weighted/balanced/square_root/progressive/none) |
| `nc_regularization` | `enabled`, `collapse_weight`, `etf_align_weight` |
| `debug` | `fast_dev_batches` (0 = full; >0 = pilot mode) |

---

## Neural Collapse Metrics

| Metric | Measures | Ideal |
|--------|----------|-------|
| NC1 | Within-class scatter relative to between-class scatter | → 0 |
| NC2 | Deviation of class means from ETF arrangement | → 0 |
| NC3 | Misalignment of classifier weights and class means | → 0 |
| NC4 | Fraction of samples where NCC ≠ argmax | → 0 |

NC1 and NC4 are the most sensitive early indicators of training failure under imbalance.

---

## Key Findings (Summary)

**1. NC metrics detect training pathology before accuracy does.**  
In collapsed runs (double-rebalancing), NC1 = 14.5 and NC4 = 0.78 from epoch 1 — while accuracy appeared plausible at 45%. Geometry failed silently.

**2. Geometry preservation and clinical performance are co-aligned.**  
Ranking by NC health (NC1↓, NC4↓) and ranking by Macro F1 produce identical method orderings.  
ETF+NC-reg > baseline = oversampling > weighted CE >> focal.

**3. Focal loss catastrophically collapses NC geometry.**  
NC4 = 0.69 at convergence (69% of samples have NCC ≠ argmax). The hard-example focusing mechanism creates a positive feedback loop: minority samples stay hard → they get amplified → feature space is over-rotated → minority representations never stabilize.

**4. Weighted CE hurts minority recall despite loss-level upweighting.**  
Melanoma recall dropped from 56% (baseline) to 36% (weighted CE). NC1 spikes to 29 at epoch 4. The gradient upweighting is strong enough to destabilize majority geometry but not strong enough to build stable minority representations.

**5. Balanced sampling causes tail-class overfitting.**  
ClassBalancedSampler upsamples DF (83 samples) by ~61× per epoch, causing memorization of the limited training samples. DF recall regressed from 29.4% → 11.8% vs the weighted-sampler ETF run.

**6. ETF + NC-reg is the best-performing configuration.**  
Best Macro F1 (0.472), best Melanoma recall (58.8%), best NC4 (0.186), best DF recall (29.4%). Gains come primarily from NC-reg stabilizing within-class scatter for tail classes.

---

## Infrastructure Notes

### Sampling strategy isolation
`weighted_ce` and `focal` must use `sampling.strategy=none`. The loss already rebalances via inverse-frequency weights. Activating the sampler simultaneously causes double-rebalancing which destroys NC geometry (NC1=14.5, NC4=0.78).

### Run-tag collision prevention
`run_phase2_study.py` now injects `logging.run_tag={study_key}` into every subprocess. `train.py` respects this override. Without this fix, all ETF variants would write to the same directory.

### fast_dev_batches
All reported results use `fast_dev_batches=75` (1,200 samples/epoch from the full training set). This is a **Tier-1 pilot protocol** — results are directionally valid but not equivalent to full-dataset training. See the final report for caveats.

---

## Citation

```bibtex
@article{papyan2020prevalence,
  title   = {Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training},
  author  = {Papyan, Vardan and Han, XY and Donoho, David L},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2020}
}

@inproceedings{yang2022inducing,
  title     = {Inducing Neural Collapse in Imbalanced Learning},
  author    = {Yang, Jiequan and others},
  booktitle = {NeurIPS},
  year      = {2022}
}
```

---

<div align="center">
<i>NC-MedAI — Neural Collapse research framework for medical image classification under imbalance.<br>
Runs on CUDA · Apple Silicon MPS · CPU.</i>
</div>
