from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data

from .profiling import rss_mb, ForwardMemoryTracer


@dataclass
class EpochProfile:
    epoch: int
    loss: float
    val_acc: float
    test_acc: float
    test_macro_f1: float
    epoch_time_s: float
    rss_before_forward_mb: float
    rss_after_forward_mb: float
    rss_after_backward_mb: float
    rss_after_step_mb: float
    forward_snapshots: List[Dict[str, float]]


def _metrics(logits: Tensor, y: Tensor, mask: Tensor) -> Dict[str, Any]:
    pred = logits[mask].argmax(dim=1).detach().cpu().numpy()
    gold = y[mask].detach().cpu().numpy()
    return {
        "acc": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "per_class_f1": {
            i: float(v) for i, v in enumerate(f1_score(gold, pred, average=None, zero_division=0))
        },
    }


def train_full_batch(model, data: Data, epochs: int = 200, lr: float = 0.01, weight_decay: float = 5e-4, device: str = "cpu"):
    model = model.to(device)
    data = data.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {"val_acc": -1.0}
    profiles: List[EpochProfile] = []
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        tracer = ForwardMemoryTracer()
        rss_before = rss_mb()
        logits = model(data.x, data.edge_index, tracer=tracer)
        rss_after_forward = rss_mb()
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        rss_after_backward = rss_mb()
        optimizer.step()
        rss_after_step = rss_mb()

        model.eval()
        with torch.no_grad():
            logits_eval = model(data.x, data.edge_index)
            val = _metrics(logits_eval, data.y, data.val_mask)
            test = _metrics(logits_eval, data.y, data.test_mask)

        epoch_time = time.perf_counter() - epoch_start
        profile = EpochProfile(
            epoch=epoch,
            loss=float(loss.item()),
            val_acc=val["acc"],
            test_acc=test["acc"],
            test_macro_f1=test["macro_f1"],
            epoch_time_s=epoch_time,
            rss_before_forward_mb=rss_before,
            rss_after_forward_mb=rss_after_forward,
            rss_after_backward_mb=rss_after_backward,
            rss_after_step_mb=rss_after_step,
            forward_snapshots=tracer.snapshots,
        )
        profiles.append(profile)

        if val["acc"] > best["val_acc"]:
            best = {
                "epoch": epoch,
                "val_acc": val["acc"],
                "test_acc": test["acc"],
                "test_macro_f1": test["macro_f1"],
                "per_class_f1": test["per_class_f1"],
            }

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_acc={val['acc']:.4f} | "
                f"test_acc={test['acc']:.4f} | test_macro_f1={test['macro_f1']:.4f} | epoch_time={epoch_time:.4f}s"
            )

    total_time = time.perf_counter() - total_start
    avg_epoch_time = float(np.mean([p.epoch_time_s for p in profiles]))
    max_rss_forward = max(p.rss_after_forward_mb for p in profiles)
    max_rss_backward = max(p.rss_after_backward_mb for p in profiles)

    return {
        "best": best,
        "profiles": profiles,
        "total_time_s": total_time,
        "avg_epoch_time_s": avg_epoch_time,
        "max_rss_after_forward_mb": max_rss_forward,
        "max_rss_after_backward_mb": max_rss_backward,
    }
