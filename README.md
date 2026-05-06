# NC-MedAI — Neural Collapse-Inspired Medical AI Research Framework

<div align="center">

**A research-grade framework for robust and imbalanced medical image classification**
*using Neural Collapse theory, ETF geometry, and representation learning.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-enabled-brightgreen)](https://tensorboard.dev)
[![Platform](https://img.shields.io/badge/Platform-CUDA%20%7C%20MPS%20%7C%20CPU-informational)](https://pytorch.org)

</div>

---

## Overview

**NC-MedAI** transforms the challenge of *imbalanced medical image classification* through the lens of **Neural Collapse (NC)** theory and **Equiangular Tight Frame (ETF) geometry**.

The key insight is simple but profound:

> *A well-trained deep network doesn't just classify — it **collapses** its feature space into a perfectly symmetric, maximally separated geometric structure. Class imbalance disrupts this geometry, and disrupted geometry means disrupted minority class detection.*

This framework:
- Quantifies **exactly how much** imbalance damages feature geometry (NC1–NC4 metrics)
- Uses **geometry-aware training** to restore structure under imbalance
- Applies this to **real medical imaging datasets** (skin lesions, chest X-rays, retinal OCT)
- Produces **publication-quality** visualisations of feature evolution
- Runs seamlessly across **CUDA, Apple Silicon MPS, and CPU** from one unified codebase

---

## The Problem: Why Standard Medical AI Fails

In a dataset of 1,000 patients where only 50 have a rare disease (5%), a model that predicts "Healthy" for every patient achieves **95% accuracy** — but detects **zero cases of disease**.

Standard cross-entropy training on imbalanced data causes:

| Symptom | Cause | Consequence |
|---------|-------|------------|
| Majority-class bias | Loss dominated by frequent classes | Minority disease missed |
| Feature scatter | Minority features don't collapse | Poor class separability |
| ETF breakdown | Class means lose symmetric structure | NCC rule fails |
| Reduced NC quality | Geometry disrupted by imbalance | Generalisation suffers |

---

## Neural Collapse: The Theoretical Foundation

Neural Collapse (Papyan et al., 2020) describes four properties that emerge at the end of training on **balanced** datasets:

```
NC1  Within-class collapse:    all features h_{i,c} → class mean μ_c
NC2  ETF structure:            class means form an Equiangular Tight Frame
NC3  Weight–mean alignment:    classifier weights W_c align with μ_c
NC4  NCC agreement:            nearest-class-center rule = argmax classifier
```

**Under class imbalance**, all four properties degrade — minority class features scatter, class means lose their ETF arrangement, and minority disease detection fails.

This framework **measures, tracks, and actively counteracts** this degradation.

---

## Project Structure

```
NEURAL-COLLAPSE/
│
├── config/
│   ├── config.yaml              ← Master config (all hyperparameters)
│   ├── config_loader.py         ← YAML loader with profile + CLI override support
│   └── profiles/                ← Hardware-specific override profiles
│       ├── apple_silicon.yaml   ← MPS, safe batch/worker settings
│       ├── cuda_gpu.yaml        ← Full throughput, AMP enabled
│       └── cpu_debug.yaml       ← Smoke test defaults, lightweight mode
│
├── data/
│   ├── medical_datasets.py      ← HAM10000, ChestXRay, RetinalOCT loaders
│   ├── dataset.py               ← CIFAR-10 with controlled imbalance injection
│   ├── imbalance_sampler.py     ← Balanced / SquareRoot / Progressive samplers
│   └── preprocessing.py        ← Medical augmentation pipelines
│
├── models/
│   ├── resnet.py                ← ResNet-18 with forward_features() API
│   ├── mobilenet.py             ← MobileNetV2 with forward_features() API
│   ├── etf_classifier.py        ← Fixed ETF head (frozen buffer, not parameter)
│   ├── prototype_head.py        ← Learnable cosine prototype classifier
│   └── model_factory.py         ← build_model(cfg, method) unified factory
│
├── training/
│   ├── trainer.py               ← Full training loop: NC tracking, AMP, TensorBoard
│   ├── losses.py                ← CE / WeightedCE / FocalLoss
│   ├── nc_regularization.py     ← NCCollapseReg / ETFAlignment / SupCon / Combined
│   └── scheduler.py             ← Cosine+warmup / Step / Constant LR
│
├── evaluation/
│   ├── nc_metrics.py            ← NC1–NC4 full suite + NCMetrics dataclass
│   ├── medical_metrics.py       ← Sensitivity / Specificity / F1 / AUC / Kappa
│   ├── evaluator.py             ← evaluate_checkpoint(), extract_features()
│   └── visualize.py             ← Sweep-level comparison plots
│
├── visualization/
│   ├── tsne_visualizer.py       ← t-SNE (PCA pre-reduction, minority highlight)
│   ├── umap_visualizer.py       ← UMAP (requires umap-learn)
│   ├── feature_geometry.py      ← PCA, norm histograms, cosine heatmap, NC evolution
│   └── confusion_analysis.py   ← Confusion matrix, per-class recall with thresholds
│
├── experiments/
│   ├── run_experiment.py        ← Atomic single-run function
│   ├── sweep.py                 ← Imbalance + method sweeps (multi-seed)
│   ├── nc_tracking.py           ← NC1–NC4 per-epoch tracking + t-SNE frames
│   ├── etf_vs_linear.py         ← Head comparison: Linear vs ETF vs Prototype
│   ├── imbalance_study.py       ← Full medical imbalance ratio sweep
│   └── ablation_studies.py      ← Backbone / NC-reg weight / ETF scale / sampling
│
├── utils/
│   ├── device.py                ← Centralized device abstraction (CUDA→MPS→CPU)
│   ├── experiment_reporter.py   ← Centralized post-training artifact generator
│   ├── seed.py                  ← set_seed() — fully reproducible experiments
│   ├── metrics.py               ← Core NC1/NC2 math (standalone)
│   └── logging_utils.py         ← get_logger(), AverageMeter
│
├── checkpoints/                 ← best_model.pth + latest.pth per run (auto-created)
├── logs/                        ← TensorBoard event files (auto-created)
├── results/                     ← All experiment outputs (auto-created)
│   ├── experiment_registry.csv  ← Central index of all runs
│   └── {run_tag}/
│       ├── config_snapshot.yaml
│       ├── metrics.csv / nc_metrics.csv / class_metrics.csv
│       ├── best_results.json / run_info.json / training_summary.json
│       ├── longtail_metrics.csv / experiment_report.md
│       ├── confusion_matrix.png / per_class_recall.png
│       └── summary.json
│
├── train.py                     ← CLI: single training run
├── run_sweep.py                 ← CLI: phase 1 / phase 2 / NC / ETF / ablation sweeps
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Environment

```bash
git clone <repo-url>
cd NEURAL-COLLAPSE

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# CUDA (adjust cu121 for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon — MPS is included in standard PyTorch ≥ 2.0
pip install torch torchvision

# Optional: UMAP visualisation
pip install umap-learn
```

### 3. Dataset setup

#### HAM10000 (Skin Lesion — Recommended)
1. Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Extract so you have: `./datasets/HAM10000/HAM10000_metadata.csv` and `./datasets/HAM10000/images/`
3. Paths are pre-configured in `config/config.yaml` — no changes needed if you use the layout above.

#### Chest X-Ray (Pneumonia)
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d ./data/raw/
```

#### Retinal OCT
```bash
kaggle datasets download -d paultimothymooney/kermany2018
unzip kermany2018.zip -d ./data/raw/OCT2017/
```

---

## Quick Start

### Single training run

```bash
# HAM10000 with ETF head (recommended)
python train.py --override dataset.name=ham10000 model.head=etf training.epochs=50

# Focal loss on Chest X-Ray
python train.py --override dataset.name=chestxray training.loss=focal

# CIFAR-10 baseline at imbalance ratio 10
python train.py --override dataset.name=cifar10 training.epochs=200 dataset.imbalance_ratio=10
```

### Smoke test (fast, CPU, ~5 min)

```bash
python train.py --override \
    dataset.name=ham10000 model.head=etf \
    training.epochs=1 training.batch_size=8 \
    training.num_workers=2 debug.fast_dev_batches=10
```

### Resume from checkpoint

```bash
python train.py --resume checkpoints/etf_ham10000_s42/latest.pth \
    --override dataset.name=ham10000 model.head=etf training.epochs=50
```

### Sweeps

```bash
# Phase 1: imbalance ratio sweep (baseline method)
python run_sweep.py --phase imbalance --method baseline --plot

# Phase 2: all methods at imbalance ratio 10
python run_sweep.py --phase method --ratio 10 --plot

# Track NC geometry across epochs
python run_sweep.py --phase nc --method etf --every 5

# ETF vs Linear vs Prototype comparison
python run_sweep.py --phase etf

# Ablation studies
python run_sweep.py --phase ablation --axis all
```

### TensorBoard

```bash
tensorboard --logdir logs/
# → http://localhost:6006
# Tracks: Loss/train, Acc/train, Acc/val, LR, NC/nc1–nc3
```

---

## Hardware Profiles

The framework runs identically across CUDA GPUs, Apple Silicon, and CPU servers. Use `--profile` to apply pre-tuned hardware settings in one flag:

```bash
# Apple Silicon Mac (M1/M2/M3)
python train.py --profile apple_silicon \
    --override dataset.name=ham10000 model.head=etf training.epochs=50

# CUDA GPU server (cloud / on-prem)
python train.py --profile cuda_gpu \
    --override dataset.name=ham10000 model.head=etf training.epochs=50

# CPU-only server / CI validation
python train.py --profile cpu_debug \
    --override dataset.name=ham10000
```

Profile settings (each is a YAML overlay, CLI overrides always win):

| Setting | `apple_silicon` | `cuda_gpu` | `cpu_debug` |
|---------|:-:|:-:|:-:|
| Device | MPS | CUDA | CPU |
| Batch size | 32 | 64 | 8 |
| Workers | 4 | 8 | 2 |
| Mixed precision | off | auto (AMP) | off |
| Lightweight viz | ✅ | ❌ | ✅ |
| fast_dev_batches | — | — | 10 |

You can also force a device directly via config:

```bash
python train.py --override system.force_device=mps
```

---

## Configuration

All hyperparameters live in `config/config.yaml`. Override at runtime using dot-notation:

```bash
python train.py --override \
  dataset.name=ham10000 \
  model.backbone=resnet18 \
  model.head=etf \
  training.epochs=50 \
  training.lr=0.01 \
  training.mixed_precision=auto \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01 \
  debug.fast_dev_batches=10
```

### Key Config Sections

| Section | Key Parameters |
|---------|---------------|
| `dataset` | `name`, `num_classes`, `image_size` |
| `model` | `backbone`, `head` (linear/etf/prototype), `pretrained` |
| `training` | `epochs`, `batch_size`, `num_workers`, `lr`, `lr_schedule`, `loss`, `mixed_precision` |
| `nc_regularization` | `enabled`, `collapse_weight`, `etf_align_weight` |
| `nc_tracking` | `enabled`, `every_n_epochs` |
| `sampling` | `strategy` (weighted/balanced/square_root/progressive) |
| `analysis` | `lightweight` (skip heavy viz on CPU/MPS) |
| `visualization` | `max_embed_samples` (cap for t-SNE/UMAP) |
| `system` | `force_device`, `adaptive_memory` |
| `tracking` | `tensorboard`, `log_dir` |
| `debug` | `fast_dev_batches` (smoke test mode) |

---

## Classifier Heads

| Head | Type | NC2 at Convergence | Use Case |
|------|------|--------------------|----------|
| `linear` | Learnable `nn.Linear` | Depends on training | General baseline |
| `etf` | Fixed ETF buffer | Guaranteed → 0 | Geometry-first learning |
| `prototype` | Learnable cosine prototypes | Low, interpretable | Minority separability |

---

## Imbalance Remedies

| Method | Mechanism | Best For |
|--------|-----------|----------|
| `baseline` | Standard CE | Reference only |
| `weighted_ce` | Inverse-frequency class weights | Simple correction |
| `focal` | Down-weights easy examples | Hard minority detection |
| `oversampling` | WeightedRandomSampler | Balanced batches |
| `balanced` | ClassBalancedSampler | Equal class coverage |
| `square_root` | √freq reweighting | Middle-ground |
| `etf` | Fixed ETF head | Geometric regularisation |
| `prototype` | Learnable prototypes | Interpretable geometry |

---

## NC Regularization

Enable in config to add auxiliary geometry losses on top of the primary criterion:

```yaml
nc_regularization:
  enabled: true
  collapse_weight: 0.01    # NCCollapseRegularizer (→ NC1)
  etf_align_weight: 0.01   # ETFAlignmentLoss (→ NC2)
```

The `CombinedNCLoss` computes:
```
L_total = L_classification + λ₁·L_collapse + λ₂·L_etf_align
```

---

## Experiment Outputs

Every run automatically generates a self-contained output directory at `results/{run_tag}/`:

| File | Description |
|------|-------------|
| `config_snapshot.yaml` | Full resolved config including all overrides |
| `metrics.csv` | Single-row final metrics summary |
| `nc_metrics.csv` | NC1–NC4 per tracked epoch |
| `class_metrics.csv` | Per-class precision, recall, F1, sensitivity, specificity |
| `best_results.json` | Best epoch NC + medical metrics snapshot |
| `longtail_metrics.csv` | Head / mid / tail group recall (support-count terciles) |
| `training_summary.json` | Epoch durations, throughput (samples/sec) |
| `run_info.json` | Git commit, device backend, CUDA/MPS info, RAM, versions |
| `experiment_report.md` | Self-contained Markdown report |
| `confusion_matrix.png` | Confusion matrix (always saved, no `--visualize` required) |
| `per_class_recall.png` | Per-class recall with clinical colour thresholds |
| `results/experiment_registry.csv` | Central cross-run index (file-lock safe) |

Checkpoints are saved at `checkpoints/{run_tag}/`:
- `best_model.pth` — best validation accuracy
- `latest.pth` — always up-to-date (crash recovery)
- `epoch_N.pth` — periodic saves (every `log_every_n_epochs`)

All checkpoints use `map_location=device` on load — **fully portable across CUDA, MPS, and CPU**.

---

## Evaluation Metrics

### Medical Metrics (primary)

| Metric | Description | Clinical Importance |
|--------|-------------|---------------------|
| Sensitivity | True positive rate per class | **Critical** — missed diagnoses cost lives |
| Specificity | True negative rate per class | Reduces false alarms |
| Macro F1 | Harmonic mean (unweighted) | Minority class performance |
| ROC-AUC | Rank discrimination (OvR macro) | Threshold-independent |
| Cohen's Kappa | Agreement beyond chance | Overall reliability |

### Neural Collapse Metrics

| Metric | Measures | Perfect Value |
|--------|----------|--------------|
| NC1 | Within-class scatter / between-class scatter | 0 |
| NC2 | ETF cosine deviation | 0 |
| NC3 | Classifier weight — class mean misalignment | 0 |
| NC4 | NCC / argmax disagreement rate | 0 |

> NC3 correctly handles both standard `nn.Linear` heads (C×D) and the ETF head's transposed buffer (D×C).

---

## Visualisations

| Plot | Script | Shows |
|------|--------|-------|
| t-SNE | `visualization/tsne_visualizer.py` | Class clusters, minority separability |
| UMAP | `visualization/umap_visualizer.py` | Global ETF geometry |
| PCA | `visualization/feature_geometry.py` | Fast geometry snapshot |
| Cosine heatmap | `visualization/feature_geometry.py` | Class-mean similarity vs ETF ideal |
| NC evolution | `visualization/feature_geometry.py` | NC1–NC4 across training epochs |
| Confusion matrix | `visualization/confusion_analysis.py` | Error patterns, row-normalised |
| Per-class recall | `visualization/confusion_analysis.py` | Clinical threshold overlay |

Heavy visualisations (t-SNE, UMAP) respect `analysis.lightweight=true` and `visualization.max_embed_samples` to prevent memory issues on unified-memory systems.

---

## Research Insights

> **NC geometry degrades faster than accuracy under imbalance.**
> NC1 and NC2 rise super-linearly with imbalance ratio, while accuracy degrades
> sub-linearly — suggesting that geometric metrics are more sensitive early
> indicators of model degradation than accuracy alone.

> **ETF heads restore NC2 by construction, but not accuracy.**
> The ETF classifier guarantees NC2 → 0 at convergence regardless of class
> imbalance, yet overall accuracy can still be lower than oversampling.
> This reveals that geometry and performance are complementary, not equivalent.

> **NC regularization helps minority recall without hurting majority accuracy.**
> Adding λ·L_collapse to the training loss encourages minority-class features to
> cluster tightly — directly improving their separability and recall.

---

## Future Work

- [ ] Extend to vision transformers (ViT-S, ViT-B)
- [ ] Longitudinal NC tracking in continual learning
- [ ] NC-aware data augmentation strategies
- [ ] Multi-label medical classification (multiple conditions per image)
- [ ] Uncertainty quantification from prototype distances
- [ ] GradCAM + NC prototype overlay for explainability

---

## Citation

If you use this framework, please cite the foundational works:

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

@inproceedings{lin2017focal,
  title     = {Focal Loss for Dense Object Detection},
  author    = {Lin, Tsung-Yi and others},
  booktitle = {ICCV},
  year      = {2017}
}
```

---

<div align="center">
<i>Built as a Neural Collapse-inspired medical AI research framework.<br>
Designed for GitHub portfolio, academic presentations, and research publication extensions.<br><br>
Runs on CUDA · Apple Silicon MPS · CPU — one unified codebase.</i>
</div>
