# Research Alignment Audit — NC-MedAI
**Date:** 2026-05-09  
**Auditor:** Automated framework audit  
**Scope:** Full repository against core research goal

---

## Research Goal (Ground Truth)

> **"How does Neural Collapse geometry behave under long-tail medical imbalance,  
> and do ETF-inspired representations preserve minority-class structure better  
> than standard linear classifiers?"**

---

## 1. Components That Directly Support the Research Goal

| Component | File | Research Relevance |
|---|---|---|
| NC1–NC4 metric suite | `evaluation/nc_metrics.py` | **Core** — directly measures collapse geometry |
| ETF classifier head | `models/etf_classifier.py` | **Core** — the primary intervention being studied |
| Linear baseline head | `models/resnet.py`, `model_factory.py` | **Core** — the comparison condition |
| HAM10000 imbalance subsampling | `data/medical_datasets.py` | **Core** — controls the independent variable |
| Medical sensitivity metrics | `evaluation/medical_metrics.py` | **Core** — primary outcome measure |
| NC regularization loss | `training/nc_regularization.py` | **Core** — geometry-aware training mechanism |
| Focal loss | `training/losses.py` | **Core** — standard minority-class remedy |
| ETF vs Linear experiment | `experiments/etf_vs_linear.py` | **Core** — direct head comparison |
| Imbalance sweep experiment | `experiments/imbalance_study.py` | **Core** — imbalance progression study |
| NC tracking per epoch | `experiments/nc_tracking.py` | **Core** — temporal geometry evolution |
| Long-tail metrics | `utils/experiment_reporter.py` | **Core** — head/mid/tail recall analysis |
| Weighted sampler | `data/imbalance_sampler.py` | **Supporting** — rebalancing baseline |
| Per-epoch NC logging | `training/trainer.py` | **Supporting** — within-run geometry tracking |

---

## 2. Components That Are Dead Weight (Low Research Relevance)

| Component | File | Assessment |
|---|---|---|
| UMAP visualizer | `visualization/umap_visualizer.py` | Adds no information beyond t-SNE; requires extra dep |
| ChestXRay dataset | `data/medical_datasets.py` | Not used in any current experiment |
| RetinalOCT dataset | `data/medical_datasets.py` | Not used in any current experiment |
| CIFAR imbalance pipeline | `data/dataset.py` | Useful for controlled sanity checks only; not the research target |
| Backbone ablation | `experiments/ablation_studies.py:run_backbone_ablation` | ResNet-18 vs MobileNetV2 is an engineering question, not a NC geometry question |
| ETF scale ablation | `experiments/ablation_studies.py:run_etf_scale_ablation` | Hyper-parameter tuning; secondary to core question |
| Hardware profiles (MPS/CUDA) | `config/profiles/` | Infrastructure; not research |
| `run_sweep.py` Phase 1 (CIFAR) | `run_sweep.py` | CIFAR sweep not aligned with HAM10000 medical focus |

---

## 3. Scientifically Meaningful Experiments

### ✅ High Value (run these)
| Experiment | Why Meaningful |
|---|---|
| ETF vs Linear @ fixed imbalance | Directly tests the head-geometry hypothesis |
| Imbalance ratio sweep (1→5→10→20→50) w/ ETF | Traces NC degradation across the transition |
| NC1/NC2/NC4 per-epoch with and without NC reg | Shows whether geometry-aware training changes collapse dynamics |
| Per-class recall (Melanoma especially) @ each ratio | Primary clinical outcome |
| Rebalancing comparison (weighted CE, focal, oversample, ETF) | Identifies best minority-class remedy |
| Multi-seed ETF vs Linear | Required for any statistical claim |

### ⚠️ Partial Value (run only if time permits)
| Experiment | Why Partial |
|---|---|
| Prototype head comparison | Interesting geometry but not the thesis focus |
| NC regularization weight sweep | Useful for tuning but secondary |
| CIFAR imbalance sanity check | Good for reproducing the Papyan baseline |

### ❌ Low/No Value (skip)
| Experiment | Why Not |
|---|---|
| Backbone ablation (ResNet vs MobileNet) | Engineering question, not geometry question |
| UMAP visualization | Redundant with PCA/t-SNE |
| ChestXRay / RetinalOCT runs | No minority-class imbalance hypothesis framed for these |

---

## 4. Metric Centrality

### Central Metrics
| Metric | Why Central |
|---|---|
| **NC1** (within-class scatter) | Primary measure of representation collapse quality |
| **NC4** (NCC disagreement) | Direct test of whether geometric classifier = argmax |
| **Melanoma sensitivity** | Clinical ground truth — rarest, most dangerous class |
| **Macro F1** | Unweighted, minority-sensitive aggregate performance |
| **Mean sensitivity** | Clinical sensitivity across all disease classes |
| **Cohen's Kappa** | Agreement metric immune to class-prior bias |

### Secondary Metrics (useful but not thesis-critical)
| Metric | Role |
|---|---|
| NC2 (ETF deviation) | Diagnostic — should be near 0 with ETF head by construction |
| NC3 (weight–mean alignment) | Confirms W–μ structure, largely redundant with NC2 |
| ROC-AUC | Useful for radiologist threshold analysis |
| Overall accuracy | Easily gamed by Nevi majority; misleading alone |

### Misleading Metrics (use cautiously)
- **Accuracy (val_acc):** Dominated by Nevi class (67% of train). ETF shows lower accuracy (53–65%) than linear (72%) but this does NOT mean ETF is worse — it may simply avoid collapsing to Nevi.
- **Weighted F1:** Over-weights Nevi, hides minority degradation.

---

## 5. Visualizations That Are Actually Useful

| Visualization | Value | Priority |
|---|---|---|
| NC1 vs epoch (ETF vs Linear) | Shows geometry evolution; thesis visual | **Essential** |
| NC4 vs epoch | Shows when NCC and argmax diverge | **Essential** |
| Melanoma recall vs imbalance ratio | Primary clinical finding | **Essential** |
| NC1 vs imbalance ratio | Core geometry-imbalance relationship | **Essential** |
| Cosine heatmap (ETF vs Linear) | Shows ETF regularity vs linear disorder | **High** |
| PCA class clusters (ETF vs Linear) | Visual confirmation of collapse geometry | **High** |
| Per-class recall bar chart | Clinical interpretability | **High** |
| Confusion matrix | Error pattern analysis | **Medium** |
| t-SNE of features | Qualitative geometry insight | **Medium** |
| NC2 vs epoch | Already guaranteed near 0 for ETF; less informative | **Low** |
| UMAP | Redundant with t-SNE | **Skip** |

---

## 6. Redundant Experiments in Current Repository

| Redundancy | Evidence |
|---|---|
| ETF run repeated 8× in registry | Registry shows `etf_ham10000_s42` appearing 9 times — same seed, same config |
| No ratio variation between runs | All registry entries use natural HAM10000 distribution (ratio=1.0) |
| `baseline_ham10000_s42` run twice | Two identical runs at different timestamps |
| CIFAR sweep code present but no CIFAR results | No CIFAR entries in experiment_registry.csv |

**The core experimental matrix (ETF × multiple ratios × multiple seeds) has not been executed yet. The two conditions that exist (ETF@30epochs, Linear@30epochs, both at ratio=1.0) are a starting point, not a study.**

---

## 7. Missing Experiments (Required for Research Completeness)

### Critical Gaps
1. **Imbalance sweep is not run on HAM10000.** `imbalance_study.py` exists and has a bug (line 80: doesn't pass `imbalance_ratio` to `get_medical_dataloaders`). No results under ratios 5, 10, 20, 50.
2. **No multi-seed validation.** Both existing conditions use seed=42 only. Any claim needs ≥3 seeds.
3. **No rebalancing comparison.** No results comparing focal, weighted_ce, oversampling, ETF head side by side at the same imbalance ratio.
4. **No NC-geometry-vs-imbalance trend plot.** The experiment is designed but never executed.
5. **No NC tracking per epoch for baseline.** nc_metrics.csv shows NC per epoch for both runs, but no joint comparison plot exists.

### Important Gaps
6. NC regularization effect on minority recall (nc_reg_weight sweep on HAM10000)
7. Pretrained vs scratch comparison — pretrained assumed throughout but never validated
8. Melanoma sensitivity as a function of imbalance ratio — the single most important clinical finding

---

## Summary Assessment

| Dimension | Status |
|---|---|
| Infrastructure | **Excellent** — logging, checkpointing, metrics all functional |
| Research scaffolding | **Good** — experiment scripts exist and are well-structured |
| Actual research evidence | **Weak** — only 2 conditions, 1 seed, 1 imbalance level |
| Core hypothesis tested | **No** — ETF vs Linear exists but at natural distribution only |
| Clinical relevance | **Partial** — sensitivity computed but not studied as primary outcome |
| Reproducibility | **Not yet** — single seed only |

**The framework is production-ready. The research is not yet started in earnest.**
