import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config
from data import get_medical_dataloaders
from models import build_model
from visualization.tsne_visualizer import plot_tsne
from experiments.plot_existing_results import plot_nc_metric_vs_epoch

def main():
    print("Initializing full figure generation for Phase 2 paper artifacts...")
    
    # We will generate t-SNE for all 6 methods run in Phase 2 so you can compare them all side-by-side
    methods = {
        "baseline": ("baseline", "linear", "Baseline"),
        "focal": ("focal", "linear", "Focal Loss"),
        "weighted_ce": ("weighted_ce", "linear", "Weighted CE"),
        "oversampling": ("oversampling", "linear", "Oversampling"),
        "etf_nc_reg": ("etf", "etf", "ETF + NC-Reg"),
        "etf_nc_reg_balanced": ("etf", "etf", "Balanced ETF + NC-Reg")
    }
    
    cfg = load_config(overrides=["dataset.name=ham10000", "dataset.imbalance_ratio=10", "seed=42"])
    cfg["dataset"]["num_classes"] = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = ROOT / "results" / "phase2" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot NC Metrics Evolution (NC1, NC2, NC3, NC4 vs Epoch)
    print("\n[Part 1] Generating NC Metric Evolution Curves...")
    nc_data = {}
    for key, (method_flag, head, label) in methods.items():
        nc_csv = ROOT / "results" / "phase2" / f"{key}_ham10000_r10_s42" / "nc_metrics.csv"
        if nc_csv.exists():
            nc_data[key] = pd.read_csv(nc_csv)
    
    if nc_data:
        # We need to map our colors/labels into the format expected by plot_nc_metric_vs_epoch
        # The original plot script uses a global PALETTE dict, so we will temporarily override it
        import experiments.plot_existing_results as p_script
        p_script.PALETTE = {
            "baseline": {"color": "#888888", "marker": "", "ls": "-", "label": "Baseline"},
            "focal": {"color": "#FF9800", "marker": "", "ls": "-", "label": "Focal Loss"},
            "weighted_ce": {"color": "#2196F3", "marker": "", "ls": "-", "label": "Weighted CE"},
            "oversampling": {"color": "#4CAF50", "marker": "", "ls": "--", "label": "Oversampling"},
            "etf_nc_reg": {"color": "#E63946", "marker": "o", "ls": "-", "label": "ETF + NC-Reg"},
            "etf_nc_reg_balanced": {"color": "#9C27B0", "marker": "s", "ls": "--", "label": "Balanced ETF + NC-Reg"}
        }
        p_script.NOTE = "Phase 2: 50-epoch sweep (10:1 imbalance)"
        
        for metric in ["nc1", "nc2", "nc3", "nc4"]:
            out_path = out_dir / f"evolution_{metric}.png"
            title = f"{metric.upper()} vs Epoch (ETF Geometry Analysis)"
            ylabel = f"{metric.upper()}"
            plot_nc_metric_vs_epoch(nc_data, col=metric, ylabel=ylabel, title=title, out=out_path)
            print(f"  Saved -> {out_path.name}")
    else:
        print("  [ERROR] No nc_metrics.csv files found. Did the sweep run successfully?")

    # 2. Extract Features and Plot t-SNE
    print("\n[Part 2] Generating t-SNE Projections...")
    print("Loading HAM10000 validation dataset...")
    _, val_loader, _ = get_medical_dataloaders(cfg, seed=42, device=device)
    class_names = getattr(val_loader.dataset, "class_names", None)
    
    for key, (method_flag, head, label) in methods.items():
        print(f"  Processing {label}...")
        cfg["model"]["head"] = head
        
        # Initialize model
        model = build_model(cfg, method=method_flag).to(device)
        model.eval()
        
        # Load Phase 2 checkpoint
        ckpt_path = ROOT / "checkpoints" / "phase2" / f"{key}_ham10000_r10_s42" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"    [SKIP] Checkpoint not found at {ckpt_path.name}")
            continue
            
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        
        all_f, all_l = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                f = model.forward_features(imgs)
                all_f.append(f.cpu())
                all_l.append(lbls)
                
        feats = torch.cat(all_f)
        labels = torch.cat(all_l)
        
        tsne_path = out_dir / f"tsne_{key}.png"
        plot_tsne(feats, labels, class_names=class_names, 
                  save_path=str(tsne_path),
                  title=f"Feature Space (t-SNE) - {label}",
                  highlight_minority=True)  # This will outline the minority class in red!
        print(f"    Saved -> {tsne_path.name}")

    print(f"\nDone! All analysis figures saved to {out_dir}/")

if __name__ == "__main__":
    main()
