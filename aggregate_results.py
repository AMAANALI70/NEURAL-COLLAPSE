import json
import pandas as pd
from pathlib import Path

results_root = Path("results")

rows = []

for run_dir in results_root.glob("etf_ham10000_s*"):

    best_path = run_dir / "best_results.json"
    class_path = run_dir / "class_metrics.csv"

    if not best_path.exists() or not class_path.exists():
        continue

    with open(best_path, "r") as f:
        best = json.load(f)

    class_df = pd.read_csv(class_path)

    melanoma_row = class_df[class_df["class"] == "Melanoma"]

    melanoma_recall = None
    if len(melanoma_row) > 0:
        melanoma_recall = float(melanoma_row.iloc[0]["recall"])

    cfg_path = run_dir / "config_snapshot.yaml"

    ratio = "unknown"

    if cfg_path.exists():
        import yaml

        with open(cfg_path, "r") as f:
            cfg = yaml.unsafe_load(f)

        ratio = cfg.get("dataset", {}).get("imbalance_ratio", 1)

    rows.append({
        "run": run_dir.name,
        "seed": run_dir.name.split("_s")[-1],
        "ratio": ratio,
        "best_acc": best.get("best_val_acc"),
        "macro_f1": best.get("macro_f1"),
        "roc_auc": best.get("roc_auc"),
        "kappa": best.get("kappa"),
        "nc1": best.get("nc1"),
        "nc2": best.get("nc2"),
        "nc3": best.get("nc3"),
        "nc4": best.get("nc4"),
        "melanoma_recall": melanoma_recall
    })

df = pd.DataFrame(rows)

df = df.sort_values(["ratio", "seed"])

print("\n================ EXPERIMENT SUMMARY ================\n")
print(df)

print("\n====================================================\n")

out_path = "aggregated_results.csv"
df.to_csv(out_path, index=False)

print(f"Saved aggregated table → {out_path}")
