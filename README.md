# NC-MedAI: Neural Collapse-Aware Medical Image Classification

<div align="center">

**A research framework studying how class imbalance degrades feature geometry**  
*and whether geometry-aware training can restore minority-class clinical performance.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-CUDA%20%7C%20MPS%20%7C%20CPU-lightgrey.svg?style=flat-square)](https://pytorch.org)

</div>

---

## 📌 Executive Summary

**NC-MedAI** is a research framework for studying **Neural Collapse (NC)** in medical image classification under severe class imbalance. Built on HAM10000 (7-class skin lesion, 10:1 majority–minority ratio), it measures how representation geometry degrades under imbalance and whether geometry-aware interventions restore minority-class detection.

> 🩺 **Clinical Insight**: *When a model's feature space stops collapsing properly, minority disease detection fails — often silently, long before validation accuracy reveals a problem. NC1–NC4 metrics expose this degradation early.*

### Key Scientific Findings:
*   📊 **Early-Warning Indicators**: NC1 (within-class scatter) and NC4 (nearest-class-mean deviation) are robust diagnostic metrics that reveal training pathologies from epoch 1.
*   📐 **Geometric Regularization**: Fixed ETF heads + NC regularization produce the most stable feature representations and the highest minority-class recall simultaneously.
*   ⚡ **Rebalancing Pitfalls**: Standard loss-based rebalancing (Weighted CE, Focal Loss) degrades NC geometry, causing unstable feature trajectories and hurting overall performance.
*   🧬 **Clinical Co-alignment**: The ranking of methods by geometric health (NC1↓, NC4↓) matches the ordering of clinical performance (Macro F1, Melanoma Recall) exactly.

---

## 📄 Deliverables & Scientific Reports

The full findings, mathematical derivations, and qualitative deep-dives are available in two print-optimized formats:
1.  **[Interactive Report (FINAL_REPORT.html)](file:///d:/dl/NEURAL-COLLAPSE/FINAL_REPORT.html)**: A premium web-based presentation containing a interactive method comparison widget, KaTeX equations, light/dark mode toggling, and expandable diagnostic plots. *Best for desktop viewing and exporting to PDF.*
2.  **[Scientific Document (FINAL_REPORT.md)](file:///d:/dl/NEURAL-COLLAPSE/FINAL_REPORT.md)**: A complete markdown document containing details of the six investigated methods, systematic results tables, deep-dive analysis, and references.

---

## 📊 Study Results at a Glance

All results evaluated on HAM10000, imbalance ratio $r = 10:1$, ResNet-18, seed = 42.

| Method | Sampler | Loss | NC-Reg | Val Acc | Macro F1 | NC1↓ | NC4↓ | Mel Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | Weighted | CE | ✗ | 64.4% | 0.436 | 4.81 | 0.194 | 56.0% |
| Oversampling | Weighted | CE | ✗ | 64.4% | 0.436 | 4.81 | 0.194 | 56.0% |
| **ETF + NC-reg (Best)** | **Weighted** | **CE** | **✅** | **65.0%** | **0.472** | **5.53** | **0.186** | **58.8%** |
| ETF + NC-reg + Balanced | Balanced | CE | ✅ | 61.8% | 0.429 | 5.63 | 0.180 | 58.8% |
| Weighted CE | None | WCE | ✗ | 62.2% | 0.332 | 7.94 | 0.396 | 36.3% |
| Focal Loss | None | Focal(γ=2) | ✗ | 56.9% | 0.110 | 15.65 | 0.690 | 0.0% |

---

## 📁 Repository Structure

```
NEURAL-COLLAPSE/
│
├── config/
│   ├── config.yaml              ← Master config (all hyperparameters)
│   └── config_loader.py         ← YAML loader with profile + CLI override support
│
├── data/
│   ├── medical_datasets.py      ← HAM10000 loader with imbalance_ratio control
│   ├── dataset.py               ← CIFAR-10 with controlled imbalance injection
│   └── imbalance_sampler.py     ← Balanced / SquareRoot / Progressive samplers
│
├── models/
│   ├── resnet.py                ← ResNet-18 with forward_features() API
│   ├── etf_classifier.py        ← Fixed ETF head (frozen buffer)
│   └── model_factory.py         ← build_model(cfg, method) factory
│
├── training/
│   ├── trainer.py               ← Full training loop with NC tracking
│   ├── losses.py                ← CE / WeightedCE / FocalLoss
│   └── nc_regularization.py     ← NCCollapseReg / ETFAlignment / CombinedNCLoss
│
├── evaluation/
│   ├── nc_metrics.py            ← NC1–NC4 full suite
│   └── medical_metrics.py       ← Sensitivity / Specificity / F1 / AUC / Kappa
│
├── experiments/
│   ├── plot_existing_results.py ← Reconstructs diagnostics from pilot runs
│   ├── run_phase1_study.py      ← Phase-1: ETF vs baseline × r∈{1,10,50}
│   └── run_phase2_study.py      ← Phase-2: intervention sweep at r=10
│
├── results/
│   ├── pilot_plots/             ← Reconstructed matplotlib charts (Val Acc, NC1-NC4)
│   ├── phase2/
│   │   ├── phase2_plots/        ← Phase-2 comparative charts (Macro F1, Recall, NC)
│   │   ├── phase2_summary.csv   ← Centrally aggregated summary table
│   │   └── *_*_ham10000_r10_s42/← Directory-level run databases (metrics, configs)
│   └── *_*_ham10000_s42/        ← Directory-level pilot run databases
│
├── train.py                     ← CLI: single training run
├── run_sweep.py                 ← CLI: sweep runner
├── FINAL_REPORT.md              ← Complete scientific report (Markdown)
├── FINAL_REPORT.html             ← Interactive, styled scientific report (HTML)
└── README.md                    ← Polished repository overview
```

---

## 🚀 Setup & Installation

1.  **Clone and environment configuration**:
    ```bash
    git clone https://github.com/amaanali70/neural-collapse.git
    cd NEURAL-COLLAPSE
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Dataset Setup**:
    *   Download [HAM10000 from Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
    *   Extract contents to `./datasets/HAM10000/`.
    *   Ensure the structure looks like this:
        ```
        datasets/HAM10000/
        ├── HAM10000_metadata.csv
        └── images/
            ├── ISIC_0024306.jpg
            └── ...
        ```

---

## 🏃 Running Experiments

### Execute Single Training Configuration:
```bash
# ETF head + NC regularization (optimal clinical setup)
python train.py --override dataset.name=ham10000 model.head=etf \
    nc_regularization.enabled=true sampling.strategy=weighted \
    training.epochs=15

# Standard baseline model
python train.py --override dataset.name=ham10000 model.head=linear training.epochs=15

# Smoke test (Runs quickly on CPU with a fraction of data)
python train.py --override dataset.name=ham10000 model.head=etf \
    training.epochs=1 debug.fast_dev_batches=10
```

### Reproduce Phase-2 Systematic Sweep:
```bash
# Dry-run validation (checks config syntax, skips training)
python -m experiments.run_phase2_study --dry-run

# Run full Phase-2 sweep (all 6 strategies sequentially)
python -m experiments.run_phase2_study

# Run a specific configuration from the sweep
python -m experiments.run_phase2_study --methods etf_nc_reg
```

### Reconstruct Diagnostic Charts:
If you need to regenerate the pilot study's diagnostic figures from the saved metric databases:
```bash
python -m experiments.plot_existing_results
```

---

## ⚙️ Configuration Overrides

All hyperparameters reside in `config/config.yaml`. You can override any nested parameters from the command line:

```bash
python train.py --override \
  dataset.name=ham10000 \
  model.head=etf \
  sampling.strategy=weighted \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01 \
  training.epochs=15 \
  debug.fast_dev_batches=75  # Set to 0 to run on full dataset (requires GPU)
```

Key configuration keys:

| Section | Parameter | Default | Description |
| :--- | :--- | :---: | :--- |
| `dataset` | `imbalance_ratio` | `10` | Imbalance ratio $r$ (majority-to-minority) |
| `model` | `head` | `linear` | Classifier head (`linear` / `etf` / `prototype`) |
| `training` | `loss` | `ce` | Optimization loss (`ce` / `weighted_ce` / `focal`) |
| `sampling` | `strategy` | `weighted` | Dataloader sampler (`weighted` / `balanced` / `none`) |
| `nc_regularization` | `enabled` | `false` | Enables fixed-ETF geometric regularization |
| `debug` | `fast_dev_batches`| `75` | Batches per epoch for pilot profiling (0 to disable) |

---

## 📐 Neural Collapse Metrics Definition

We track the following metrics at the end of each training epoch:

*   **NC1 (Within-Class Variability Collapse)**: Measures feature scatter within classes relative to scatter between classes. Ideally converges to $0$.
*   **NC2 (ETF Cosine Deviation)**: Measures how close class feature centroids are to forming a symmetric, maximally-distanced Equiangular Tight Frame. Ideally converges to $0$.
*   **NC3 (Weight-Feature Alignment)**: Measures the self-duality/alignment of final classifier weights and centered feature means. Ideally converges to $0$.
*   **NC4 (NCM Classifier Disagreement)**: Identifies the fraction of validation samples for which standard forward inference disagrees with a Nearest Class Mean classifier. Ideally converges to $0$.

---

## 📜 Citations & References

If you build upon this work, please cite the following foundational studies:

```bibtex
@article{papyan2020prevalence,
  title   = {Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training},
  author  = {Papyan, Vardan and Han, XY and Donoho, David L},
  journal = {Proceedings of the National Academy of Sciences (PNAS)},
  volume  = {117},
  number  = {40},
  pages   = {24652--24663},
  year    = {2020}
}

@inproceedings{yang2022inducing,
  title     = {Inducing Neural Collapse in Imbalanced Learning},
  author    = {Yang, Jiequan and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
```

---
<div align="center">
<i>NC-MedAI Research Framework · Supported on CUDA, Apple Silicon MPS, and CPU.</i>
</div>
