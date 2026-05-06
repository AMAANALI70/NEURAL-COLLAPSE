"""
training/trainer.py  (updated)
─────────────────────────────────────────────────────────────────────────────
Extended Trainer with:
  • NC-aware regularization (CombinedNCLoss)
  • Per-epoch NC1–NC4 tracking
  • TensorBoard logging (optional)
  • Progressive sampler epoch update
  • Medical dataset compatible
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.logging_utils import AverageMeter, get_logger
from evaluation.nc_metrics import compute_all_nc_metrics, NCMetrics
from training.losses import get_criterion
from training.scheduler import build_scheduler
from training.nc_regularization import CombinedNCLoss


class Trainer:
    """
    Full training + evaluation loop for medical AI / NC experiments.

    New vs original
    ---------------
    • Supports CombinedNCLoss (classification + NC regularization).
    • Logs NC1–NC4 every `track_nc_every_n_epochs` epochs.
    • Writes to TensorBoard SummaryWriter when cfg.tracking.tensorboard=true.
    • Calls sampler.set_epoch(e) for ProgressiveSampler compatibility.

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader
    val_loader   : DataLoader
    class_weights: torch.Tensor
    cfg          : dict
    method       : str
    seed         : int
    device       : torch.device
    run_tag      : str
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        cfg: dict,
        method: str = "baseline",
        seed: int = 42,
        device: Optional[torch.device] = None,
        run_tag: str = "run",
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.method       = method
        self.seed         = seed
        self.run_tag      = run_tag

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model.to(self.device)

        # ── Logging ───────────────────────────────────────────────────────────
        log_dir = cfg["logging"].get("results_dir", "./results")
        self.logger = get_logger(name=f"trainer.{run_tag}", log_dir=log_dir)

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.writer = None
        if cfg.get("tracking", {}).get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = Path(cfg.get("tracking", {}).get("log_dir", "./logs")) / run_tag
                self.writer = SummaryWriter(log_dir=str(tb_dir))
                self.logger.info(f"  TensorBoard → {tb_dir}")
            except ImportError:
                self.logger.warning("tensorboard not installed; skipping TB logging.")

        # ── Loss ──────────────────────────────────────────────────────────────
        base_crit = get_criterion(method, class_weights, cfg, self.device)
        nc_cfg    = cfg.get("nc_regularization", {})
        if nc_cfg.get("enabled", False):
            self.criterion = CombinedNCLoss(
                base_criterion   = base_crit,
                num_classes      = cfg["dataset"]["num_classes"],
                collapse_weight  = nc_cfg.get("collapse_weight", 0.01),
                etf_align_weight = nc_cfg.get("etf_align_weight", 0.01),
            )
            self._use_nc_loss = True
            self.logger.info("  NC regularization ENABLED")
        else:
            self.criterion    = base_crit
            self._use_nc_loss = False

        # ── Optimiser ─────────────────────────────────────────────────────────
        train_cfg = cfg["training"]
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(train_cfg["lr"]),
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            nesterov=True,
        )
        self.scheduler  = build_scheduler(self.optimizer, cfg)
        self.epochs     = int(train_cfg["epochs"])
        self.num_classes = cfg["dataset"]["num_classes"]

        # ── NC tracking ───────────────────────────────────────────────────────
        self.track_nc_every = cfg.get("evaluation", {}).get(
            "track_nc_every_n_epochs", 0)
        self.nc_history: List[Dict] = []

        # ── Checkpointing ─────────────────────────────────────────────────────
        self.save_ckpt = cfg["logging"].get("save_checkpoints", True)
        self.ckpt_dir  = Path(cfg["logging"].get("checkpoint_dir", "./checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = cfg["logging"].get("log_every_n_epochs", 5)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the full training pipeline and return results dict."""
        history: Dict[str, list] = {
            "train_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc = 0.0
        best_state   = None

        self.logger.info(
            f"[{self.run_tag}] device={self.device} "
            f"method={self.method} epochs={self.epochs}"
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Update progressive sampler if applicable
            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_loss, train_acc = self._train_epoch(epoch)
            val_acc               = self._eval_epoch()
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone()
                                for k, v in self.model.state_dict().items()}

            # ── TensorBoard ───────────────────────────────────────────────────
            if self.writer:
                self.writer.add_scalar("Loss/train",    train_loss, epoch)
                self.writer.add_scalar("Acc/train",     train_acc,  epoch)
                self.writer.add_scalar("Acc/val",       val_acc,    epoch)
                self.writer.add_scalar(
                    "LR", self.optimizer.param_groups[0]["lr"], epoch)

            # ── NC Tracking ───────────────────────────────────────────────────
            if self.track_nc_every > 0 and (
                epoch % self.track_nc_every == 0 or epoch == self.epochs
            ):
                nc = self._compute_nc(epoch)
                self.nc_history.append(nc.to_dict())
                if self.writer:
                    self.writer.add_scalar("NC/nc1", nc.nc1, epoch)
                    self.writer.add_scalar("NC/nc2", nc.nc2, epoch)
                    if not (nc.nc3 != nc.nc3):   # not nan
                        self.writer.add_scalar("NC/nc3", nc.nc3, epoch)

            # ── Console logging ───────────────────────────────────────────────
            if epoch % self.log_every == 0 or epoch == self.epochs:
                elapsed = time.time() - t0
                self.logger.info(
                    f"  ep {epoch:3d}/{self.epochs} "
                    f"| loss {train_loss:.4f} "
                    f"| tr {train_acc:.1f}% "
                    f"| val {val_acc:.1f}% "
                    f"| best {best_val_acc:.1f}% "
                    f"| {elapsed:.1f}s"
                )

        # ── Save best checkpoint ──────────────────────────────────────────────
        if self.save_ckpt and best_state is not None:
            ckpt = self.ckpt_dir / f"{self.run_tag}_best.pt"
            torch.save({"state_dict": best_state, "best_val_acc": best_val_acc,
                        "nc_history": self.nc_history}, ckpt)
            self.logger.info(f"  Checkpoint → {ckpt}")

        # ── Final NC metrics on best model ────────────────────────────────────
        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})
        final_nc = self._compute_nc(epoch=self.epochs)
        self.logger.info(
            f"  Final: NC1={final_nc.nc1:.4f} NC2={final_nc.nc2:.6f} "
            f"NC3={final_nc.nc3:.4f} NC4={final_nc.nc4:.4f} "
            f"best_val={best_val_acc:.2f}%"
        )

        if self.writer:
            self.writer.close()

        return {
            **history,
            "best_val_acc": best_val_acc,
            "nc1": final_nc.nc1, "nc2": final_nc.nc2,
            "nc3": final_nc.nc3, "nc4": final_nc.nc4,
            "nc_history": self.nc_history,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int):
        self.model.train()
        loss_m = AverageMeter("Loss")
        acc_m  = AverageMeter("Acc")

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            features = self.model.forward_features(images)
            logits   = self.model.fc(features)

            if self._use_nc_loss:
                loss = self.criterion(logits, features, labels)
            else:
                loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            preds = logits.argmax(1)
            acc   = (preds == labels).float().mean().item() * 100.0
            loss_m.update(loss.item(), images.size(0))
            acc_m.update(acc,          images.size(0))

        return loss_m.avg, acc_m.avg

    @torch.no_grad()
    def _eval_epoch(self) -> float:
        self.model.eval()
        correct = total = 0
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            preds  = self.model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        return 100.0 * correct / total if total else 0.0

    @torch.no_grad()
    def _compute_nc(self, epoch: int = 0) -> NCMetrics:
        self.model.eval()
        all_f, all_l, all_g = [], [], []
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            feats  = self.model.forward_features(images)
            logits = self.model.fc(feats)
            all_f.append(feats.cpu())
            all_l.append(labels)
            all_g.append(logits.cpu())

        feats  = torch.cat(all_f)
        labels = torch.cat(all_l)
        logits = torch.cat(all_g)

        return compute_all_nc_metrics(
            feats, labels, self.num_classes,
            model=self.model, logits=logits, epoch=epoch,
        )
