# Technical Report: Neural Collapse Under Class Imbalance
### DL_PROJ — Deep Learning Research Project

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation and Problem Statement](#2-motivation-and-problem-statement)
3. [Theoretical Background](#3-theoretical-background)
4. [Dataset Design](#4-dataset-design)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Imbalance Remedy Methods](#7-imbalance-remedy-methods)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Experiment Phases](#9-experiment-phases)
10. [Project Structure & Module Reference](#10-project-structure--module-reference)
11. [Configuration System](#11-configuration-system)
12. [Results and Analysis](#12-results-and-analysis)
13. [Limitations and Future Work](#13-limitations-and-future-work)
14. [References](#14-references)

---

## 1. Executive Summary

This project investigates **Neural Collapse (NC)** — a geometric phenomenon observed in the terminal phase of deep neural network training — under conditions of **class imbalance**. Using CIFAR-10 as the benchmark dataset and ResNet-18 as the primary backbone, the project systematically sweeps six imbalance ratios (1×–100×) and compares five remediation strategies: baseline cross-entropy, class-weighted loss, focal loss, oversampling, and an Equiangular Tight Frame (ETF) classifier.

The codebase is organized as a production-grade modular Python project with clean separation of concerns across `config`, `data`, `models`, `training`, `evaluation`, `experiments`, and `utils` packages. Experiments are reproducible (multi-seed), configurable via a single YAML file, and launchable from two CLI entry-points (`train.py`, `run_sweep.py`).

**Key findings (expected):**
- Class imbalance monotonically degrades both NC1/NC2 geometry and classification accuracy.
- ETF classifiers restore NC2 geometry most aggressively, even under severe imbalance.
- Oversampling achieves the best accuracy recovery among standard remedies.
- Focal loss and weighted CE offer intermediate improvements over the baseline.

---

## 2. Motivation and Problem Statement

### 2.1 Why Neural Collapse Matters

Deep classifiers trained to convergence exhibit a surprisingly structured terminal state: all within-class feature vectors collapse to a single point (their class mean), and all class means arrange themselves into a perfectly symmetric simplex — an Equiangular Tight Frame. This phenomenon, termed **Neural Collapse** by Papyan et al. (2020), suggests that the learned geometry of feature space is not arbitrary — it converges to an optimal, theoretically predictable configuration.

Understanding NC has practical implications:
- **Robustness**: NC correlates with better out-of-distribution generalisation.
- **Transfer**: ETF-structured features transfer more uniformly across classes.
- **Interpretability**: The geometry provides a principled measure of training quality.

### 2.2 The Class Imbalance Problem

Real-world datasets are rarely balanced. Medical imaging, fraud detection, and long-tail recognition benchmarks routinely exhibit majority:minority class ratios of 10:1 to 1000:1. Standard cross-entropy training on imbalanced data causes the classifier to:
- Over-predict majority classes.
- Produce features that cluster tightly for majority classes but diffusely for minority classes.
- **Disrupt the NC geometry**, preventing the clean collapse observed on balanced data.

### 2.3 Research Questions

1. How does increasing class imbalance affect NC1 and NC2 metrics quantitatively?
2. Which imbalance remedy best restores NC geometry alongside accuracy?
3. Does restoring NC geometry causally improve classification accuracy?
4. Can a fixed ETF head force NC2 alignment even without balanced data?

---

## 3. Theoretical Background

### 3.1 Neural Collapse (NC)

Formally, NC is defined over the terminal training phase (after training loss approaches zero) by four properties:

| Property | Description |
|----------|-------------|
| **NC1** | Within-class variability collapses: all features `h_{i,c}` → class mean `μ_c` |
| **NC2** | Class means form an ETF: equal norms, equal pairwise angles |
| **NC3** | The linear classifier weight vectors `W` align with the class means |
| **NC4** | Decision rule simplifies to nearest class-mean (maximum cosine similarity) |

This project measures **NC1** and **NC2** directly from penultimate-layer features.

### 3.2 NC1 — Within-Class Scatter

NC1 is measured as the ratio of within-class scatter to between-class scatter:

```
NC1 = trace(Σ_W) / trace(Σ_B)
```

Where:
- `Σ_W = (1/N) Σ_c Σ_{i∈c} (h_i - μ_c)(h_i - μ_c)^T`  (within-class covariance)
- `Σ_B = (1/N) Σ_c n_c (μ_c - μ_G)(μ_c - μ_G)^T`  (between-class covariance)
- `μ_G` is the global feature mean

**Perfect collapse → NC1 = 0.** Under imbalance, minority-class features scatter widely, inflating NC1.

### 3.3 NC2 — ETF Alignment

An ETF satisfies:
```
(μ_c - μ_G) / ‖μ_c - μ_G‖  forms a simplex ETF
```

The ideal cosine similarity between any two distinct class means is `-1 / (C - 1)`.

NC2 is measured as the mean squared deviation from this ideal:
```
NC2 = mean_{i≠j} [ cos(μ_i, μ_j) - (-1/(C-1)) ]²
```

**Perfect ETF alignment → NC2 = 0.** Under imbalance, majority-class means dominate the geometry, pushing minority means out of the ETF arrangement.

### 3.4 Equiangular Tight Frame (ETF) Classifier

An ETF classifier replaces the learnable linear head `W ∈ R^{D×C}` with a **fixed** weight matrix that satisfies the ETF condition by construction:

```
W^T W = C/(C-1) · (I - (1/C) · 11^T)
```

The backbone is trained to align its features to this fixed geometry, which theoretically guarantees NC2 = 0 at convergence regardless of class frequencies in the training data. This idea originates from Zhu et al. (2021) and Yang et al. (2022).

---

## 4. Dataset Design

### 4.1 CIFAR-10 Baseline

CIFAR-10 consists of 60,000 32×32 colour images across 10 balanced classes (6,000 images/class). The standard split is 50,000 training / 10,000 test images. It is used here because:
- It is small enough for rapid experimental iteration.
- Its balanced nature makes it easy to introduce controlled imbalance.
- The ResNet-18 baseline achieves ~93% accuracy, providing a strong upper bound.

### 4.2 Controlled Long-Tail Imbalance

The `ImbalancedCIFAR10` class (in `data/dataset.py`) constructs a long-tail version:

- **Class 0** (airplane) retains all 5,000 training samples → the **majority class**.
- **Classes 1–9** are sub-sampled such that:

```
n_minority = floor(n_majority / imbalance_ratio)
```

| Ratio | n_majority | n_minority | Total training samples |
|-------|-----------|------------|----------------------|
| 1 | 5,000 | 5,000 | 50,000 |
| 5 | 5,000 | 1,000 | 14,000 |
| 10 | 5,000 | 500 | 9,500 |
| 20 | 5,000 | 250 | 7,250 |
| 50 | 5,000 | 100 | 5,900 |
| 100 | 5,000 | 50 | 5,450 |

Sub-sampling uses a fixed NumPy RNG seed (0) for reproducible splits independent of the training seed.

### 4.3 Data Augmentation

| Split | Transforms |
|-------|-----------|
| Train | RandomCrop(32, padding=4) → RandomHorizontalFlip → ToTensor → Normalize |
| Val/Test | ToTensor → Normalize |

Normalization constants: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010).

### 4.4 Class Weights

For each training dataset, per-class inverse-frequency weights are computed:
```
w_c = 1 / (freq_c + ε),   normalised so Σ w_c = 1
```
These are exposed by `get_dataloaders()` and consumed by the weighted CE and focal loss criteria.

---

## 5. Model Architecture

### 5.1 ResNet-18 (Primary Backbone)

Standard ResNet-18 is designed for 224×224 ImageNet images. For CIFAR-10 (32×32), two modifications are made in `models/resnet.py`:

1. **First convolution**: 7×7 stride-2 → **3×3 stride-1** (preserves spatial resolution).
2. **Max-pool removed**: replaced with `nn.Identity()` to avoid over-downsampling.

The model exposes:
- `forward_features(x)` → penultimate representation of shape `(B, 512)`
- `forward(x)` → logits of shape `(B, 10)`
- `feature_dim = 512`

### 5.2 MobileNetV2 (Ablation Backbone)

A lightweight alternative in `models/mobilenet.py`. The first inverted residual block's stride is reduced from 2 to 1 for CIFAR compatibility. Feature dimension is 1280.

### 5.3 ETF Classifier Head

Defined in `models/etf_classifier.py`. The weight matrix `W ∈ R^{512×10}` is:
1. Initialised via QR decomposition of a random Gaussian matrix.
2. Column-centred and column-normalised.
3. Registered as a **buffer** (not a parameter) → never updated by the optimiser.

Forward pass: `logits = scale · normalize(features) @ W`

The `build_model()` factory in `models/model_factory.py` swaps the standard `nn.Linear` head with the ETF head when `method="etf"`.

---

## 6. Training Pipeline

### 6.1 Optimiser

SGD with Nesterov momentum:
- Learning rate: 0.1
- Momentum: 0.9
- Weight decay: 5×10⁻⁴
- Nesterov: True

### 6.2 Learning Rate Schedule

| Schedule | Behaviour |
|----------|-----------|
| `cosine` | Linear warm-up (5 epochs) → CosineAnnealingLR to η_min=1e-6 |
| `step` | ×0.1 at epochs 100 and 150 |
| `none` | Constant LR |

### 6.3 Trainer Class

`training/trainer.py` implements the full training loop:

```
for epoch in 1..T:
    train_epoch()   → updates model weights, records loss/acc
    eval_epoch()    → computes val accuracy
    scheduler.step()
    if val_acc > best: save checkpoint

compute_nc()        → extracts val features, computes NC1/NC2
```

**Checkpoint format**: `{run_tag}_best.pt` containing `state_dict` + `best_val_acc`.

### 6.4 AverageMeter

Lightweight running-statistics tracker used inside `_train_epoch()` to accumulate loss and accuracy without storing all batch values.

---

## 7. Imbalance Remedy Methods

### 7.1 Baseline
Standard cross-entropy with no imbalance correction. Serves as the lower bound for all comparisons.

### 7.2 Weighted Cross-Entropy
```
L = - Σ_i  w_{y_i} · log p_{y_i}
```
Weights are the inverse class frequencies computed from the training set. Automatically up-weights rare classes without changing the data distribution.

### 7.3 Focal Loss
```
L_focal = - α_t · (1 - p_t)^γ · log(p_t)
```
- `γ = 2.0` (default): smoothly down-weights easy examples.
- `α_t`: per-class weight (auto-computed from class frequencies when `alpha: null` in config).

Implemented in `training/losses.py` as `FocalLoss(nn.Module)`. Works on top of log-softmax outputs for numerical stability.

### 7.4 Oversampling
Uses PyTorch's `WeightedRandomSampler` to draw training batches so that each class appears with equal probability. The dataset remains unchanged; only the sampling strategy changes.

### 7.5 ETF Classifier
Backbone is trained with standard CE loss, but the classifier head is the fixed ETF matrix. The backbone must learn to produce features aligned to this fixed geometry. This implicitly enforces NC2 = 0 at optimality.

---

## 8. Evaluation Framework

### 8.1 Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Val Accuracy | correct / total × 100 | Classification performance |
| NC1 | trace(Σ_W) / trace(Σ_B) | Within-class scatter (↓ better) |
| NC2 | mean squared cosine deviation from ETF ideal | ETF alignment (↓ better) |

### 8.2 Feature Extraction

After training, `extract_features()` in `evaluation/evaluator.py` passes the full validation set through `model.forward_features()` to obtain `(N, D)` feature tensors. NC metrics are then computed on these tensors by `utils/metrics.py`.

### 8.3 Visualisations

Three plot types are generated by `evaluation/visualize.py`:

| Plot | X-axis | Y-axis | Purpose |
|------|--------|--------|---------|
| `plot_imbalance_sweep` | Imbalance ratio | Acc / NC1 / NC2 | Phase 1 trend analysis |
| `plot_method_comparison` | Method | Acc / NC1 / NC2 | Phase 2 bar chart |
| `plot_nc_scatter` | NC1 (log) | Accuracy | Geometry vs performance |

All plots use a headless `Agg` backend (no display required), 150 DPI, and are saved as `.png` to `results/`.

### 8.4 Multi-Seed Aggregation

The sweep functions in `experiments/sweep.py` run each (method, ratio) combination across all seeds in `cfg.seeds` (default: [42, 123, 7]), then report:
- `acc_mean`, `acc_std`
- `nc1_mean`, `nc1_std`
- `nc2_mean`, `nc2_std`

Results are saved as CSVs alongside the plots.

---

## 9. Experiment Phases

### Phase 1 — Concept Exploration
**Goal**: Establish baseline behaviour and verify NC metrics are working.

**Steps**:
1. Train balanced (ratio=1) ResNet-18 for 200 epochs.
2. Confirm NC1 ≈ 0 and NC2 ≈ 0 at convergence (~93% val acc).
3. Sweep ratios {1, 5, 10, 20, 50, 100} with baseline CE.
4. Plot Accuracy, NC1, NC2 vs ratio.

**Command**:
```bash
python run_sweep.py --phase imbalance --method baseline --plot
```

**Expected observations**:
- Accuracy drops monotonically from ~93% to ~70–75% at ratio=100.
- NC1 rises monotonically (minority classes scatter more).
- NC2 rises monotonically (class means diverge from ETF arrangement).

---

### Phase 2 — Experimental Validation
**Goal**: Compare all five remedies at a fixed ratio (10 or 100).

**Steps**:
1. Run all five methods × 3 seeds at imbalance_ratio=10.
2. Aggregate statistics and generate comparison bar charts.
3. Repeat at ratio=100 for severe imbalance analysis.

**Command**:
```bash
python run_sweep.py --phase method --ratio 10 --plot
python run_sweep.py --phase method --ratio 100 --plot
```

**Expected ordering** (accuracy at ratio=100):
```
oversampling ≈ focal ≥ weighted > etf > baseline
```

**Expected NC2 ordering** (lower = more ETF-like):
```
etf << oversampling < weighted < focal < baseline
```

---

### Phase 3 — Deep Analysis
**Goal**: Understand the relationship between NC geometry and accuracy.

**Steps**:
1. Collect all (method, ratio, NC1, NC2, accuracy) data points.
2. Generate NC1 vs Accuracy scatter plot coloured by method.
3. Perform correlation analysis: does lower NC1 causally predict higher accuracy?

**Command**:
```bash
# After sweeps have run, call directly in Python:
from evaluation.visualize import plot_nc_scatter
import pandas as pd
df = pd.read_csv("results/sweep_methods_r10.csv")
plot_nc_scatter(df, save_dir="results", prefix="phase3")
```

---

### Phase 4 — Extension (Optional)
**Goal**: Ablation studies and backbone comparison.

**Options**:
- Swap to MobileNetV2: `--override model.backbone=mobilenetv2`
- ETF with frozen backbone: `--override etf.fix_backbone=true`
- Different focal gamma: `--override focal_loss.gamma=0.5`

---

## 10. Project Structure & Module Reference

```
DL_PROJ/
├── config/
│   ├── config.yaml          ← Master hyperparameter file
│   ├── config_loader.py     ← load_config() with CLI overrides
│   └── __init__.py
├── data/
│   ├── dataset.py           ← ImbalancedCIFAR10, get_dataloaders()
│   └── __init__.py
├── models/
│   ├── resnet.py            ← CIFAR ResNet-18 (forward_features exposed)
│   ├── mobilenet.py         ← CIFAR MobileNetV2
│   ├── etf_classifier.py    ← Fixed ETF head (buffer, not parameter)
│   ├── model_factory.py     ← build_model(cfg, method)
│   └── __init__.py
├── training/
│   ├── losses.py            ← CrossEntropyLoss, WeightedCE, FocalLoss
│   ├── scheduler.py         ← CosineAnnealingLR, StepLR, constant
│   ├── trainer.py           ← Trainer class (train → eval → NC → ckpt)
│   └── __init__.py
├── evaluation/
│   ├── evaluator.py         ← evaluate_checkpoint(), extract_features()
│   ├── visualize.py         ← 3 plot functions (sweep, bar, scatter)
│   └── __init__.py
├── experiments/
│   ├── run_experiment.py    ← run_single_experiment() — atomic unit
│   ├── sweep.py             ← run_imbalance_sweep(), run_method_sweep()
│   └── __init__.py
├── utils/
│   ├── seed.py              ← set_seed() (random, numpy, torch, cudnn)
│   ├── metrics.py           ← compute_nc_metrics() → {nc1, nc2}
│   ├── logging_utils.py     ← get_logger(), AverageMeter
│   └── __init__.py
├── notebooks/               ← Exploratory Jupyter notebooks
├── results/                 ← Auto-created output directory
│   ├── checkpoints/         ← Best model weights per run
│   ├── *.csv                ← Sweep result tables
│   └── *.png                ← Plots
├── train.py                 ← CLI: single experiment
├── run_sweep.py             ← CLI: full sweeps
├── requirements.txt
└── README.md
```

### Call Graph (simplified)

```
train.py
  └── run_single_experiment()
        ├── set_seed()
        ├── get_dataloaders()        → ImbalancedCIFAR10
        ├── build_model()            → ResNet18 / ETFClassifier
        └── Trainer.run()
              ├── get_criterion()    → CE / FocalLoss
              ├── build_scheduler()  → CosineAnnealingLR
              ├── _train_epoch()     → AverageMeter
              ├── _eval_epoch()      → accuracy
              └── _compute_nc()     → compute_nc_metrics()

run_sweep.py
  ├── run_imbalance_sweep()
  │     └── run_single_experiment() × (|ratios| × |seeds|)
  └── run_method_sweep()
        └── run_single_experiment() × (|methods| × |seeds|)
```

---

## 11. Configuration System

All hyperparameters are centralised in `config/config.yaml`. The loader (`config_loader.py`) supports runtime overrides via dot-notation strings.

### Full Config Reference

```yaml
seeds: [42, 123, 7]

dataset:
  name: CIFAR10
  root: ./data/raw
  num_classes: 10
  imbalance_ratios: [1, 5, 10, 20, 50, 100]

model:
  backbone: resnet18        # resnet18 | mobilenetv2
  pretrained: false
  feature_dim: 512

training:
  epochs: 200
  batch_size: 128
  optimizer: SGD
  lr: 0.1
  momentum: 0.9
  weight_decay: 5.0e-4
  lr_schedule: cosine       # cosine | step | none
  warmup_epochs: 5

focal_loss:
  gamma: 2.0
  alpha: null               # null → auto from class frequencies

etf:
  enabled: false
  fix_backbone: false

evaluation:
  nc_batch_size: 256

logging:
  results_dir: ./results
  save_checkpoints: true
  checkpoint_dir: ./results/checkpoints
  log_every_n_epochs: 10

sweep:
  methods: [baseline, weighted_loss, focal_loss, oversampling, etf]
```

### CLI Override Examples

```bash
# Reduce epochs for quick testing
python train.py --method baseline --ratio 1 --override training.epochs=10

# Switch backbone
python train.py --method baseline --ratio 10 --override model.backbone=mobilenetv2

# Tune focal gamma
python train.py --method focal_loss --ratio 20 --override focal_loss.gamma=0.5
```

---

## 12. Results and Analysis

### 12.1 Expected Phase 1 Results (Baseline Sweep)

| Ratio | Val Acc (%) | NC1 | NC2 |
|-------|------------|-----|-----|
| 1 | ~93.0 | ~0.01 | ~0.001 |
| 5 | ~88.5 | ~0.08 | ~0.012 |
| 10 | ~84.0 | ~0.18 | ~0.028 |
| 20 | ~80.0 | ~0.35 | ~0.055 |
| 50 | ~75.5 | ~0.72 | ~0.120 |
| 100 | ~71.0 | ~1.45 | ~0.230 |

*NC1 and NC2 both rise super-linearly with imbalance ratio, while accuracy degrades sub-linearly — suggesting that geometry degrades faster than performance, and that NC metrics are more sensitive early indicators of imbalance damage.*

### 12.2 Expected Phase 2 Results (Methods at Ratio=100)

| Method | Val Acc (%) | NC1 | NC2 |
|--------|------------|-----|-----|
| Baseline | ~71.0 | ~1.45 | ~0.230 |
| Weighted CE | ~79.5 | ~0.90 | ~0.140 |
| Focal Loss | ~81.0 | ~0.85 | ~0.130 |
| Oversampling | ~82.5 | ~0.70 | ~0.090 |
| ETF | ~79.0 | ~0.55 | ~0.015 |

**Key takeaway**: The ETF head achieves the lowest NC2 by construction (the head forces ETF alignment), but its accuracy is lower than oversampling because it constrains the feature geometry without addressing the data imbalance. Oversampling achieves the best accuracy by ensuring the backbone sees balanced batches.

### 12.3 NC1 vs Accuracy Scatter Interpretation

When plotted on a log-NC1 vs Accuracy scatter:
- All methods form a roughly monotonic negative correlation.
- ETF points cluster at the left (low NC1) regardless of accuracy — confirming it improves geometry without always improving accuracy.
- Oversampling points cluster at the top-right (high accuracy, moderate NC1).
- The baseline forms a clean decreasing curve from balanced (top-left) to severely imbalanced (bottom-right).

---

## 13. Limitations and Future Work

### Limitations

| Issue | Description |
|-------|-------------|
| Single dataset | Results are specific to CIFAR-10; long-tail benchmarks (ImageNet-LT, Places-LT) may differ |
| Fixed backbone | ResNet-18 may not represent behaviour of larger models (ViT, ResNet-50) |
| NC3/NC4 not measured | Classifier weight alignment and decision rule collapse are not tracked |
| Wall-clock time | Full 3-seed sweep at all ratios × all methods ≈ 10–20 GPU hours |
| No mixup/cutmix | Modern augmentation strategies that may further restore NC are not explored |

### Future Directions

1. **Larger backbones**: Study NC under imbalance for ResNet-50, ViT-S.
2. **NC3/NC4 tracking**: Measure `cos(W_c, μ_c)` alignment during training.
3. **Long-tail benchmarks**: Apply pipeline to ImageNet-LT and CIFAR-100-LT.
4. **Combination methods**: ETF head + oversampling + mixup together.
5. **Online NC monitoring**: Track NC1/NC2 every epoch (not just at the end) to understand the training dynamics.
6. **Continual learning**: Investigate how NC geometry evolves as new classes are introduced sequentially.

---

## 14. References

| # | Citation |
|---|---------|
| 1 | Papyan, V., Han, X. Y., & Donoho, D. L. (2020). *Prevalence of neural collapse during the terminal phase of deep learning training.* PNAS. |
| 2 | Zhu, Z., Ding, T., Zhou, J., et al. (2021). *A geometric analysis of neural collapse with unconstrained features.* NeurIPS. |
| 3 | Yang, J., Shi, R., Tang, C., et al. (2022). *Inducing neural collapse in imbalanced learning: Do we really need a learnable classifier at the end of deep neural network?* NeurIPS. |
| 4 | Lin, T.-Y., Goyal, P., Girshick, R., et al. (2017). *Focal loss for dense object detection.* ICCV. |
| 5 | He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR. |
| 6 | Cao, K., Wei, C., Gaidon, A., et al. (2019). *Learning imbalanced datasets with label-distribution-aware margin loss.* NeurIPS. |
| 7 | Sandler, M., Howard, A., Zhu, M., et al. (2018). *MobileNetV2: Inverted residuals and linear bottlenecks.* CVPR. |

---

*Report generated for DL_PROJ — Neural Collapse Under Class Imbalance.*
*All code is located in `d:\dl\DL_PROJ\`.*
