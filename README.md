# NC-MedAI — Neural Collapse-Inspired Medical AI Research Framework

<div align="center">

**A research-grade framework for robust and imbalanced medical image classification**
*using Neural Collapse theory, ETF geometry, and representation learning.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-enabled-brightgreen)](https://tensorboard.dev)

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
DL_PROJ/
│
├── config/
│   ├── config.yaml              ← Master config (dataset, model, training, NC)
│   └── config_loader.py         ← YAML loader with CLI dot-notation overrides
│
├── data/
│   ├── medical_datasets.py      ← HAM10000, ChestXRay, RetinalOCT loaders
│   ├── dataset.py               ← CIFAR-10 with controlled imbalance
│   ├── imbalance_sampler.py     ← Balanced / SquareRoot / Progressive samplers
│   └── preprocessing.py        ← Medical augmentation pipelines
│
├── models/
│   ├── resnet.py                ← CIFAR/medical ResNet-18 (feature layer exposed)
│   ├── mobilenet.py             ← CIFAR/medical MobileNetV2
│   ├── etf_classifier.py        ← Fixed ETF head (frozen buffer, not parameter)
│   ├── prototype_head.py        ← Learnable cosine prototype classifier
│   └── model_factory.py         ← build_model(cfg, method) unified factory
│
├── training/
│   ├── trainer.py               ← Trainer: NC tracking, TensorBoard, NC loss
│   ├── losses.py                ← CE / WeightedCE / FocalLoss
│   ├── nc_regularization.py     ← NCCollapseReg / ETFAlignment / SupCon / Combined
│   └── scheduler.py             ← Cosine+warmup / Step / Constant LR
│
├── evaluation/
│   ├── nc_metrics.py            ← NC1–NC4 full suite + NCMetrics dataclass
│   ├── medical_metrics.py       ← Sensitivity / Specificity / F1 / AUC / Kappa
│   ├── evaluator.py             ← evaluate_checkpoint(), extract_features()
│   └── visualize.py             ← Imbalance sweep / method comparison plots
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
│   ├── seed.py                  ← set_seed() — fully reproducible experiments
│   ├── metrics.py               ← Core NC1/NC2 math (standalone)
│   └── logging_utils.py         ← get_logger(), AverageMeter
│
├── logs/                        ← TensorBoard event files
├── checkpoints/                 ← Best model weights per run
├── results/                     ← CSVs, plots, sweep outputs
├── notebooks/                   ← Exploratory Jupyter notebooks
│
├── train.py                     ← CLI: single training run
├── run_sweep.py                 ← CLI: phase 1 / phase 2 sweeps
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Environment

```bash
git clone <repo-url>
cd DL_PROJ

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# For CUDA (adjust cu121 for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: UMAP visualisation
pip install umap-learn
```

### 3. Dataset setup

#### HAM10000 (Skin Lesion — Recommended)
1. Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Extract to `./data/raw/HAM10000/`
3. Update `config/config.yaml`:
```yaml
medical:
  ham10000:
    csv_path: ./data/raw/HAM10000/HAM10000_metadata.csv
    img_dir:  ./data/raw/HAM10000/images
```

#### Chest X-Ray (Pneumonia)
```bash
# Kaggle CLI
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
# Focal loss on HAM10000
python train.py --dataset ham10000 --method focal --seed 42

# ETF head on Chest X-Ray
python train.py --dataset chestxray --method etf --seed 42

# CIFAR-10 baseline at imbalance ratio 10
python train.py --dataset cifar10 --method baseline --ratio 10 --seed 42

# Override any config value
python train.py --method focal --override training.epochs=100 training.lr=0.005
```

### Track NC metrics across epochs

```bash
# Log NC1–NC4 every 5 epochs + save t-SNE frames
python -m experiments.nc_tracking --method etf --every 5 --tsne

# Watch in TensorBoard
tensorboard --logdir ./logs
```

### Compare ETF vs Linear vs Prototype head

```bash
python -m experiments.etf_vs_linear --ratio 10
```

### Ablation studies

```bash
# All ablation axes
python -m experiments.ablation_studies --axis all

# Just backbone comparison
python -m experiments.ablation_studies --axis backbone

# NC regularization weight sweep
python -m experiments.ablation_studies --axis nc_reg
```

### Full sweep

```bash
# Phase 1: imbalance ratio sweep (baseline method)
python run_sweep.py --phase imbalance --method baseline --plot

# Phase 2: all methods at ratio 10
python run_sweep.py --phase method --ratio 10 --plot
```

---

## Configuration

All hyperparameters live in `config/config.yaml`. Override at runtime:

```bash
python train.py --override \
  dataset.name=ham10000 \
  model.backbone=resnet18 \
  training.epochs=50 \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01
```

### Key Config Sections

| Section | Key Parameters |
|---------|---------------|
| `dataset` | `name`, `num_classes`, `image_size` |
| `model` | `backbone`, `head` (linear/etf/prototype), `pretrained` |
| `training` | `epochs`, `lr`, `lr_schedule`, `loss` |
| `nc_regularization` | `enabled`, `collapse_weight`, `etf_align_weight` |
| `sampling` | `strategy` (weighted/balanced/square_root/progressive) |
| `evaluation` | `track_nc_every_n_epochs` |
| `tracking` | `tensorboard`, `log_dir` |

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

## Evaluation Metrics

### Medical Metrics (primary)

| Metric | Description | Clinical Importance |
|--------|-------------|---------------------|
| Sensitivity | True positive rate | **Critical** — misses cost lives |
| Specificity | True negative rate | Reduces false alarms |
| Macro F1 | Harmonic mean (unweighted) | Minority class performance |
| ROC-AUC | Rank discrimination | Threshold-independent |
| Cohen's Kappa | Agreement beyond chance | Overall reliability |

### Neural Collapse Metrics

| Metric | Measures | Perfect Value |
|--------|----------|--------------|
| NC1 | Within-class scatter / between-class scatter | 0 |
| NC2 | ETF cosine deviation | 0 |
| NC3 | Classifier weight — class mean misalignment | 0 |
| NC4 | NCC / argmax disagreement rate | 0 |

---

## Visualisations

| Plot | Script | Shows |
|------|--------|-------|
| t-SNE | `visualization/tsne_visualizer.py` | Class clusters, minority separability |
| UMAP | `visualization/umap_visualizer.py` | Global ETF geometry |
| PCA | `visualization/feature_geometry.py` | Fast geometry snapshot |
| Cosine heatmap | `visualization/feature_geometry.py` | Class-mean similarity vs ETF ideal |
| NC evolution | `visualization/feature_geometry.py` | NC1–NC4 across training epochs |
| Feature norms | `visualization/feature_geometry.py` | Per-class collapse quality |
| Confusion matrix | `visualization/confusion_analysis.py` | Error patterns, row-normalised |
| Per-class recall | `visualization/confusion_analysis.py` | Clinical threshold overlay |

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
Designed for GitHub portfolio, academic presentations, and research publication extensions.</i>
</div>
