# NC-MedAI: Neural Collapse-Aware Medical Image Classification

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat-square)](https://github.com/amaanali70/neural-collapse)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg?style=flat-square)](https://github.com/amaanali70/neural-collapse)
[![Tech Stack](https://img.shields.io/badge/tech--stack-PyTorch%20%7C%20Python-informational.svg?style=flat-square)](https://pytorch.org)

**NC-MedAI** is a deep learning research framework designed to study the mathematical phenomenon of **Neural Collapse (NC)** in medical image classification under heavy class imbalance. By analyzing last-layer feature representations on the HAM10000 skin lesion dataset, NC-MedAI quantifies how class imbalance degrades representation geometry and evaluates geometry-aware training methods to restore minority disease detection.

### The Problem It Solves
Standard convolutional networks trained on highly imbalanced medical datasets often achieve high overall accuracy while failing catastrophically to detect rare, critical classes (e.g., Melanoma). 
*   **Representation Warp**: Imbalance forces the feature space to warp, squeezing minority classes into narrow regions dominated by majority class boundaries.
*   **Rebalancing Inefficiency**: Traditional rebalancing techniques (like Focal Loss or Weighted Cross-Entropy) alter training gradients, which can corrupt the geometry of the penultimate feature layer and cause model instability.
*   **Early Warning**: NC-MedAI monitors the four properties of Neural Collapse (NC1–NC4) as early-warning indicators, showing how representation geometry degrades long before accuracy drop occurs. It provides a fixed Equiangular Tight Frame (ETF) head + Neural Collapse regularization to stabilize feature learning and improve clinical recall.

---

## Table of Contents
1. [Features](#features)
2. [Demo & Screenshots](#demo--screenshots)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
7. [API Documentation (CLI Interface)](#api-documentation-cli-interface)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Contributing](#contributing)
11. [Roadmap / Future Improvements](#roadmap--future-improvements)
12. [License](#license)
13. [Contact / Author](#contact--author)

---

## Features
*   **NC1–NC4 Metric Tracking**: Real-time evaluation of within-class variability collapse (NC1), Equiangular Tight Frame deviation (NC2), classifier weight alignment (NC3), and Nearest Class Mean disagreement (NC4).
*   **Fixed ETF Classifier Head**: Replaces standard learnable linear heads with a rigid, non-learnable Equiangular Tight Frame structure to enforce maximum geometric class separation.
*   **Neural Collapse Regularization Loss**: Implements custom regularization functions ($\mathcal{L}_{collapse}$ and $\mathcal{L}_{align}$) to force penultimate features into the pre-allocated ETF vectors.
*   **Sweep Runner Framework**: Automation scripts to run, log, and evaluate baseline and intervention runs sequentially across different imbalance ratios.
*   **Interactive Visual Dashboard**: A print-optimized, interactive HTML report with a dynamic performance simulator widget.

---

## Demo / Screenshots (Placeholder Section)

### Visual Interactive Dashboard
The research report features an interactive web interface where you can simulate and compare metrics across different methods:
```
+-----------------------------------------------------------------+
|  [Interactive Performance Simulator]                           |
|                                                                 |
|  Select Method:                         Metrics Highlight:      |
|  (o) ETF + NC-reg                      * Val Acc:  65.0%        |
|  ( ) Linear Baseline                   * Macro F1: 0.4720       |
|  ( ) Focal Loss                        * NC1:      5.53         |
|                                                                 |
|  Clinical Recall Highlights:                                    |
|  Melanoma:        [====================>        ] 58.8%        |
|  Dermatofibroma:  [===========>                 ] 29.4%        |
+-----------------------------------------------------------------+
```
*   To view the interactive dashboard locally, open [FINAL_REPORT.html](file:///d:/dl/NEURAL-COLLAPSE/FINAL_REPORT.html) in your browser.
*   To view the generated diagnostic curves, refer to the matplotlib plots stored under `results/pilot_plots/` and `results/phase2/phase2_plots/`.

---

## Tech Stack

### Frontend
*   **Core**: HTML5, Vanilla JavaScript (ES6+ for interactive widgets)
*   **Styling**: Vanilla CSS3 (featuring responsive grids, CSS variables, dark/light modes, and print-optimized sheets)
*   **Libraries**: KaTeX CDN (for mathematical notation rendering)

### Backend
*   **Languages**: Python 3.10+
*   **Core Framework**: PyTorch 2.0+ (with CUDA/MPS acceleration support)
*   **Data Science**: NumPy, Pandas, Scikit-learn
*   **Visualization**: Matplotlib

### Database
*   **Storage**: Flat-file database logging (run metrics saved as `.csv` and run parameters stored in `.json` formatting)

### Tools & DevOps
*   **Environments**: Virtualenv / Pip package manager
*   **Configuration**: YAML-based config loader with hierarchical overrides

---

## Project Structure
```
NEURAL-COLLAPSE/
├── config/                  # Configuration files and YAML loaders
│   ├── config.yaml          # Master configuration file
│   └── config_loader.py     # Hierarchical configuration loader
├── data/                    # Dataset loaders and sampling strategies
│   ├── medical_datasets.py  # HAM10000 data loader with imbalance ratio control
│   └── imbalance_sampler.py # Weighted and balanced mini-batch samplers
├── models/                  # Neural network architectures
│   ├── resnet.py            # ResNet-18 feature-extraction architecture
│   └── etf_classifier.py    # Fixed ETF head implementation
├── training/                # Optimization loops and training utilities
│   ├── trainer.py           # Core training loop with epoch-level metrics
│   ├── losses.py            # Focal Loss, Weighted CE, and standard CE functions
│   └── nc_regularization.py # Custom Neural Collapse loss regularizers
├── evaluation/              # Validation metrics
│   ├── nc_metrics.py        # Mathematical implementations of NC1–NC4
│   └── medical_metrics.py   # Clinical sensitivity, specificity, and Macro F1
├── experiments/             # High-level sweep execution files
│   ├── plot_existing_results.py # Diagnostic plot generator script
│   └── run_phase2_study.py  # Automated baseline/intervention sweep script
├── results/                 # Output directories
│   ├── pilot_plots/         # Matplotlib diagnostic charts
│   └── phase2/              # Systematic sweep metrics database
├── train.py                 # Core CLI entrypoint to train models
├── FINAL_REPORT.md          # Scientific report in Markdown
├── FINAL_REPORT.html        # Interactive report in HTML
└── README.md                # Project documentation
```

---

## Installation & Setup

### Prerequisites
*   Python 3.10 or higher
*   pip package manager
*   CUDA-compatible GPU or Apple Silicon Mac (highly recommended)

### Step-by-Step Installation Instructions
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/amaanali70/neural-collapse.git
    cd NEURAL-COLLAPSE
    ```
2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```
3.  **Activate the Environment**:
    *   **Windows**:
        ```powershell
        .\venv\Scripts\activate
        ```
    *   **Linux/macOS**:
        ```bash
        source venv/bin/activate
        ```
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables Setup
A template is provided in `.env.example`. Create a `.env` file in the root directory if environment-specific variables are required:
```bash
cp .env.example .env
```

### How to Run Locally
Ensure you have downloaded the [HAM10000 Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and extracted it into the `./datasets/HAM10000/` directory:
```
datasets/HAM10000/
├── HAM10000_metadata.csv
└── images/
    ├── ISIC_0024306.jpg
    └── ...
```

---

## Usage

### Train a Single Model Configuration
Run standard cross-entropy training on the HAM10000 dataset:
```bash
python train.py --override dataset.name=ham10000 model.head=linear training.epochs=15
```

### Induce Neural Collapse (ETF Head + NC Regularization)
Train with a fixed geometric classifier head and Neural Collapse losses:
```bash
python train.py --override \
  dataset.name=ham10000 \
  model.head=etf \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01 \
  sampling.strategy=weighted \
  training.epochs=15
```

---

## API Documentation (CLI Interface)

The primary entry point is `train.py`. Model configuration overrides are handled via the `--override` flag.

### CLI Config Options
| Parameter Override | Options / Type | Description |
| :--- | :--- | :--- |
| `dataset.name` | `ham10000` / `cifar10` | The dataset to train on |
| `dataset.imbalance_ratio` | `int` (default: `10`) | Ratio of majority to minority classes |
| `model.head` | `linear` / `etf` | Classifier head architecture type |
| `training.loss` | `ce` / `weighted_ce` / `focal` | Optimization loss function |
| `sampling.strategy` | `weighted` / `balanced` / `none` | Dataloader sampler strategy |
| `nc_regularization.enabled` | `true` / `false` | Enables Neural Collapse regularization |

### Example CLI Overrides
```json
// Example: Training Weighted CE Loss with a learnable head and no sampling rebalance
python train.py --override dataset.name=ham10000 training.loss=weighted_ce sampling.strategy=none
```

---

## Testing

To verify the code installation and model training configurations, you can run a quick CPU "smoke test" by limiting the number of mini-batches:

```bash
# Smoke test (runs 1 epoch with 10 mini-batches on CPU)
python train.py --override dataset.name=ham10000 model.head=etf training.epochs=1 debug.fast_dev_batches=10
```

To run and verify the diagnostic plotting scripts:
```bash
python -m experiments.plot_existing_results
```

---

## Deployment

NC-MedAI is a research codebase. However, training runs and summaries can be compiled and deployed locally.

### Report Distribution
The metrics dashboard compiles to a single, zero-dependency HTML file. You can deploy it by:
1.  Compiling HTML assets.
2.  Deploying `FINAL_REPORT.html` directly to static site hosts (GitHub Pages, Netlify, or Vercel).
3.  Generating a print version by opening `FINAL_REPORT.html` in Chrome/Edge, opening the print settings (`Ctrl+P`), selecting **Save as PDF**, and sharing the document.

---

## Contributing

We welcome contributions to expand this geometric analysis framework.

### Contribution Guidelines
1.  **Code Consistency**: Preserve the existing modular design pattern: datasets in `data/`, metrics in `evaluation/`, and loss models in `training/`.
2.  **Document Integrity**: Maintain docstrings and annotations for mathematical functions. Ensure equations align with Papyan et al. (2020) notations.
3.  **Pull Request Process**:
    *   Fork the repository and create a new feature branch.
    *   Ensure the CPU smoke test passes (`debug.fast_dev_batches=10`).
    *   Provide a clear summary of how your proposed change affects NC1–NC4 feature metrics.

---

## Roadmap / Future Improvements
*   [ ] **Model Scaling**: Integrate larger pre-trained medical backbones (e.g., ConvNeXt, Swin Transformers).
*   [ ] **Dynamic Loss Schedulers**: Research dynamic schedules where the fixed ETF constraint is introduced gradually as training progresses.
*   [ ] **Multi-Dataset Benchmarks**: Expand evaluation to Chest X-ray (NIH) and Histopathology (PCam) datasets to check generalizability.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact / Author

*   **NC-MedAI Research Group** - [amaanali70](https://github.com/amaanali70)
*   Project Link: [https://github.com/amaanali70/neural-collapse](https://github.com/amaanali70/neural-collapse)
