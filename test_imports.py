"""test_imports.py — Run from inside DL_PROJ/"""
import sys
sys.path.insert(0, ".")

errors = []
checks = 0

def ok(msg):
    global checks
    checks += 1
    print(f"  [OK]   {msg}")

def fail(label, exc):
    errors.append(f"  [FAIL] {label}: {exc}")
    print(errors[-1])

# 1 — Config
try:
    from config import load_config
    cfg = load_config()
    ok("config.load_config")
except Exception as e: fail("config", e)

# 2 — Utils
try:
    from utils import set_seed, get_logger
    from utils.metrics import compute_nc_metrics
    ok("utils")
except Exception as e: fail("utils", e)

# 3 — ResNet18
try:
    import torch
    from models import ResNet18
    m   = ResNet18(num_classes=10)
    x   = torch.randn(2, 3, 32, 32)
    f   = m.forward_features(x)
    g   = m(x)
    assert f.shape == (2, 512) and g.shape == (2, 10)
    ok("models.ResNet18 forward_features + forward")
except Exception as e: fail("ResNet18", e)

# 4 — ETFClassifier
try:
    import torch
    from models import ETFClassifier
    etf = ETFClassifier(512, 10, scale=16.0)
    out = etf(torch.randn(4, 512))
    assert out.shape == (4, 10)
    ok("models.ETFClassifier")
except Exception as e: fail("ETFClassifier", e)

# 5 — PrototypeHead
try:
    import torch
    from models import PrototypeHead
    ph  = PrototypeHead(512, 7)
    out = ph(torch.randn(4, 512))
    assert out.shape == (4, 7)
    ok("models.PrototypeHead")
except Exception as e: fail("PrototypeHead", e)

# 6 — NC Metrics NC1–NC4
try:
    import torch
    from evaluation.nc_metrics import compute_all_nc_metrics
    feats  = torch.randn(100, 512)
    labels = torch.randint(0, 7, (100,))
    nc     = compute_all_nc_metrics(feats, labels, num_classes=7)
    assert hasattr(nc, "nc1") and hasattr(nc, "nc2")
    ok(f"evaluation.nc_metrics  NC1={nc.nc1:.4f}  NC2={nc.nc2:.6f}")
except Exception as e: fail("nc_metrics", e)

# 7 — NC Regularization
try:
    import torch
    from training.nc_regularization import NCCollapseRegularizer, ETFAlignmentLoss, CombinedNCLoss
    reg  = NCCollapseRegularizer(weight=0.01)
    feats = torch.randn(16, 512, requires_grad=True)
    lbls  = torch.randint(0, 7, (16,))
    loss  = reg(feats, lbls)
    assert loss.item() >= 0
    ok("training.nc_regularization")
except Exception as e: fail("nc_regularization", e)

# 8 — FocalLoss
try:
    import torch
    from training.losses import FocalLoss
    fl     = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 7)
    labels = torch.randint(0, 7, (8,))
    loss   = fl(logits, labels)
    assert loss.item() >= 0
    ok("training.losses.FocalLoss")
except Exception as e: fail("FocalLoss", e)

# 9 — Samplers
try:
    import numpy as np
    from data.imbalance_sampler import ClassBalancedSampler, SquareRootSampler, ProgressiveSampler
    targets = np.random.choice(7, size=200).tolist()
    s   = ClassBalancedSampler(targets)
    idx = list(iter(s))
    assert len(idx) > 0
    ok(f"data.imbalance_sampler  (balanced n={len(idx)})")
except Exception as e: fail("imbalance_sampler", e)

# 10 — Visualization exports
try:
    from visualization import (
        plot_tsne, plot_umap, plot_pca,
        plot_feature_norms, plot_cosine_heatmap,
        plot_nc_evolution, plot_confusion_matrix,
        plot_per_class_recall,
    )
    ok("visualization (all 8 exports)")
except Exception as e: fail("visualization imports", e)

# 11 — Medical metrics
try:
    import numpy as np
    from evaluation.medical_metrics import compute_medical_metrics
    y_t = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_p = np.array([0, 1, 2, 2, 0, 0, 2, 3])
    m   = compute_medical_metrics(y_t, y_p, num_classes=4)
    assert "macro_f1" in m and "sensitivity" in m and "kappa" in m
    kappa = m["kappa"]
    f1    = m["macro_f1"]
    ok(f"evaluation.medical_metrics  kappa={kappa:.3f}  F1={f1:.3f}")
except Exception as e: fail("medical_metrics", e)

# 12 — Preprocessing transforms
try:
    from data.preprocessing import get_medical_transforms, get_cifar_transforms
    _ = get_medical_transforms("train", 224, "ham10000")
    _ = get_medical_transforms("val",   224, "chestxray")
    _ = get_cifar_transforms("train")
    ok("data.preprocessing (all transforms)")
except Exception as e: fail("preprocessing", e)

# 13 — Scheduler
try:
    import torch
    import torch.optim as optim
    from training.scheduler import build_scheduler
    from config import load_config
    cfg2 = load_config()
    net  = __import__("models").ResNet18(10)
    opt  = optim.SGD(net.parameters(), lr=0.1)
    sch  = build_scheduler(opt, cfg2)
    sch.step()
    ok("training.scheduler.build_scheduler")
except Exception as e: fail("scheduler", e)

# 14 — build_model ETF
try:
    import torch
    from models import build_model
    from config import load_config
    cfg2 = load_config()
    net  = build_model(cfg2, method="etf")
    x    = torch.randn(2, 3, 32, 32)
    out  = net(x)
    ok(f"models.build_model(etf)  output={tuple(out.shape)}")
except Exception as e: fail("build_model(etf)", e)

# 15 — build_model Prototype
try:
    import torch
    from models import build_model
    from config import load_config
    cfg2 = load_config()
    net  = build_model(cfg2, method="prototype")
    x    = torch.randn(2, 3, 32, 32)
    out  = net(x)
    ok(f"models.build_model(prototype)  output={tuple(out.shape)}")
except Exception as e: fail("build_model(prototype)", e)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print(f"  Passed : {checks} / 15")
if errors:
    print(f"  Failed : {len(errors)}")
    for e in errors:
        print(f"    {e}")
else:
    print("  All checks passed — framework is import-clean.")
print("=" * 50)
