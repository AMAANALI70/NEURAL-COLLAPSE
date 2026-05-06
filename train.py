"""
train.py — CLI entry point for a single training run.
Updated for the NC-MedAI Medical AI Framework.

Usage
-----
# HAM10000 with focal loss
python train.py --dataset ham10000 --method focal --seed 42

# CIFAR-10 baseline at imbalance ratio 10
python train.py --dataset cifar10 --method baseline --ratio 10

# ETF head on chest X-ray
python train.py --dataset chestxray --method etf --seed 42

# NC regularization enabled via override
python train.py --method focal --override nc_regularization.enabled=true

# Short test run
python train.py --method baseline --override training.epochs=3 dataset.name=cifar10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from data import get_dataloaders, get_medical_dataloaders
from models import build_model
from training import Trainer, get_criterion
from evaluation.nc_metrics import compute_all_nc_metrics
from evaluation.medical_metrics import compute_medical_metrics, print_medical_metrics
from visualization.tsne_visualizer import plot_tsne
from visualization.feature_geometry import plot_cosine_heatmap, plot_pca
from visualization.confusion_analysis import plot_confusion_matrix, plot_per_class_recall
from utils.seed import set_seed
from utils.logging_utils import get_logger
import torch
import numpy as np

_logger = get_logger("train")

_METHOD_MAP = {
    "weighted_loss": "weighted",
    "weighted_ce":   "weighted",
    "focal_loss":    "focal",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NC-MedAI: Single experiment training run."
    )
    p.add_argument("--config",  default=None,
                   help="Path to YAML config (default: config/config.yaml)")
    p.add_argument("--dataset", default=None,
                   choices=["cifar10", "ham10000", "chestxray", "retinal_oct"],
                   help="Dataset override (sets dataset.name in config)")
    p.add_argument("--method",  default="baseline",
                   choices=["baseline", "weighted_ce", "focal", "oversampling",
                             "etf", "prototype", "balanced", "square_root"],
                   help="Training method / classifier head")
    p.add_argument("--ratio",   type=int, default=None,
                   help="Imbalance ratio (CIFAR-10 only)")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--visualize", action="store_true",
                   help="Generate t-SNE, PCA, cosine heatmap, confusion plots after training")
    p.add_argument("--override", nargs="*", default=None, metavar="KEY=VALUE",
                   help="Dot-notation config overrides")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build overrides list ──────────────────────────────────────────────────
    overrides = list(args.override or [])
    if args.dataset:
        overrides.append(f"dataset.name={args.dataset}")
    if args.ratio is not None:
        overrides.append(f"dataset.imbalance_ratio={args.ratio}")

    cfg    = load_config(config_path=args.config, overrides=overrides or None)
    method = _METHOD_MAP.get(args.method, args.method)
    set_seed(args.seed)

    dataset_name = cfg["dataset"]["name"].lower()
    num_classes  = cfg["dataset"]["num_classes"]
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imb_ratio    = args.ratio or 1

    _logger.info(f"Dataset={dataset_name}  Method={method}  Seed={args.seed}  Device={device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if dataset_name == "cifar10":
        train_loader, val_loader, class_weights = get_dataloaders(
            cfg, imbalance_ratio=imb_ratio, method=method, seed=args.seed)
        class_names = None
    else:
        train_loader, val_loader, class_weights = get_medical_dataloaders(cfg, seed=args.seed)
        class_names = getattr(train_loader.dataset, "class_names", None)

    # ── Model + Trainer ───────────────────────────────────────────────────────
    model   = build_model(cfg, method=method)
    run_tag = f"{method}_{dataset_name}_s{args.seed}"

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        class_weights=class_weights, cfg=cfg, method=method,
        seed=args.seed, device=device, run_tag=run_tag,
    )
    results = trainer.run()

    # ── Full medical evaluation ───────────────────────────────────────────────
    model.eval()
    all_f, all_l, all_g = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            f    = model.forward_features(imgs)
            g    = model.fc(f)
            all_f.append(f.cpu()); all_l.append(lbls); all_g.append(g.cpu())

    feats  = torch.cat(all_f)
    labels = torch.cat(all_l)
    logits = torch.cat(all_g)

    y_true = labels.numpy()
    y_pred = logits.argmax(1).numpy()
    y_prob = torch.softmax(logits, 1).numpy()

    nc  = compute_all_nc_metrics(feats, labels, num_classes, model, logits)
    med = compute_medical_metrics(y_true, y_pred, y_prob,
                                  class_names=class_names, num_classes=num_classes)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print(f"  NC-MedAI Training Complete")
    print(f"  Dataset : {dataset_name}   Method : {method}")
    print(f"  Seed    : {args.seed}      Device : {device}")
    print("─"*60)
    print(f"  Best Val Accuracy : {results['best_val_acc']:.2f}%")
    print(f"  Macro F1          : {med['macro_f1']:.4f}")
    print(f"  Mean Sensitivity  : {med['mean_sensitivity']:.4f}")
    print(f"  Mean Specificity  : {med['mean_specificity']:.4f}")
    print(f"  Cohen Kappa       : {med['kappa']:.4f}")
    print(f"  ROC-AUC           : {med['roc_auc']:.4f}")
    print("─"*60)
    print(f"  NC1 (scatter)     : {nc.nc1:.6f}")
    print(f"  NC2 (ETF dev)     : {nc.nc2:.8f}")
    print(f"  NC3 (W–μ align)   : {nc.nc3:.6f}")
    print(f"  NC4 (NCC disagree): {nc.nc4:.6f}")
    print("═"*60 + "\n")

    # ── Optional visualisations ───────────────────────────────────────────────
    if args.visualize:
        vis_dir = str(Path(cfg["logging"]["results_dir"]) / run_tag)
        _logger.info("Generating visualisations …")

        plot_tsne(feats, labels, class_names=class_names,
                  save_path=f"{vis_dir}/tsne.png",
                  title=f"t-SNE — {method} on {dataset_name}")

        plot_pca(feats, labels, class_names=class_names,
                 save_path=f"{vis_dir}/pca.png",
                 title=f"PCA — {method} on {dataset_name}")

        plot_cosine_heatmap(feats, labels, num_classes, class_names=class_names,
                            save_path=f"{vis_dir}/cosine_heatmap.png")

        import numpy as _np
        from sklearn.metrics import confusion_matrix as _cm
        cm = _np.array(med["confusion_matrix"])
        plot_confusion_matrix(cm, class_names=class_names,
                              save_path=f"{vis_dir}/confusion_matrix.png")

        plot_per_class_recall(med["sensitivity"],
                              save_path=f"{vis_dir}/per_class_recall.png")
        _logger.info(f"  Visualisations saved to {vis_dir}/")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    out_dir = Path(cfg["logging"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method":       method,
        "dataset":      dataset_name,
        "seed":         args.seed,
        "best_val_acc": results["best_val_acc"],
        "macro_f1":     med["macro_f1"],
        "sensitivity":  med["mean_sensitivity"],
        "specificity":  med["mean_specificity"],
        "kappa":        med["kappa"],
        "roc_auc":      med["roc_auc"],
        "nc1": nc.nc1, "nc2": nc.nc2, "nc3": nc.nc3, "nc4": nc.nc4,
    }
    json_path = out_dir / f"{run_tag}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved → {json_path}")


if __name__ == "__main__":
    main()
