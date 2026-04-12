from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import ClusterData, ClusterLoader, GraphSAINTNodeSampler


@dataclass
class EpochProfile:
    epoch: int
    loss: float
    epoch_time_s: float
    rss_after_forward_mb: float
    rss_after_backward_mb: float
    forward_snapshots: List[Dict[str, float]]


def _get_process_rss_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def _safe_macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _safe_per_class_f1(y_true, y_pred) -> Dict[int, float]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    scores = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {int(label): float(score) for label, score in zip(labels, scores)}


@torch.no_grad()
def _evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    metrics = {}
    for split_name, mask_name in [("train", "train_mask"), ("val", "val_mask"), ("test", "test_mask")]:
        mask = getattr(data, mask_name)
        y_true = data.y[mask].cpu()
        y_pred = pred[mask].cpu()

        if y_true.numel() == 0:
            metrics[f"{split_name}_acc"] = 0.0
            metrics[f"{split_name}_macro_f1"] = 0.0
            metrics[f"{split_name}_per_class_f1"] = {}
            continue

        metrics[f"{split_name}_acc"] = float(accuracy_score(y_true, y_pred))
        metrics[f"{split_name}_macro_f1"] = float(_safe_macro_f1(y_true, y_pred))
        metrics[f"{split_name}_per_class_f1"] = _safe_per_class_f1(y_true, y_pred)

    return metrics


def _make_snapshot_collector():
    snapshots: List[Dict[str, float]] = []

    def snapshot_fn(step_name: str):
        snapshots.append({
            "step": step_name,
            "rss_mb": _get_process_rss_mb(),
        })

    return snapshots, snapshot_fn


def train_full_batch(
    model,
    data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {
        "epoch": -1,
        "val_acc": -1.0,
        "test_acc": 0.0,
        "test_macro_f1": 0.0,
        "per_class_f1": {},
    }

    profiles: List[EpochProfile] = []
    total_start = time.perf_counter()
    max_rss_after_forward_mb = 0.0
    max_rss_after_backward_mb = 0.0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        snapshots, snapshot_fn = _make_snapshot_collector()
        out = model(data.x, data.edge_index, snapshot_fn=snapshot_fn)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        rss_after_forward_mb = _get_process_rss_mb()

        loss.backward()
        rss_after_backward_mb = _get_process_rss_mb()

        optimizer.step()
        epoch_time_s = time.perf_counter() - epoch_start

        metrics = _evaluate(model, data)

        if metrics["val_acc"] > best["val_acc"]:
            best = {
                "epoch": epoch,
                "val_acc": metrics["val_acc"],
                "test_acc": metrics["test_acc"],
                "test_macro_f1": metrics["test_macro_f1"],
                "per_class_f1": metrics["test_per_class_f1"],
            }

        max_rss_after_forward_mb = max(max_rss_after_forward_mb, rss_after_forward_mb)
        max_rss_after_backward_mb = max(max_rss_after_backward_mb, rss_after_backward_mb)

        profiles.append(
            EpochProfile(
                epoch=epoch,
                loss=float(loss.item()),
                epoch_time_s=float(epoch_time_s),
                rss_after_forward_mb=float(rss_after_forward_mb),
                rss_after_backward_mb=float(rss_after_backward_mb),
                forward_snapshots=snapshots,
            )
        )

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={loss.item():.4f} | "
                f"val_acc={metrics['val_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | "
                f"test_macro_f1={metrics['test_macro_f1']:.4f} | "
                f"epoch_time={epoch_time_s:.4f}s"
            )

    total_time_s = time.perf_counter() - total_start
    avg_epoch_time_s = sum(p.epoch_time_s for p in profiles) / len(profiles)

    return {
        "best": best,
        "profiles": profiles,
        "total_time_s": total_time_s,
        "avg_epoch_time_s": avg_epoch_time_s,
        "max_rss_after_forward_mb": max_rss_after_forward_mb,
        "max_rss_after_backward_mb": max_rss_after_backward_mb,
    }


def train_graphsaint(
    model,
    data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {
        "epoch": -1,
        "val_acc": -1.0,
        "test_acc": 0.0,
        "test_macro_f1": 0.0,
        "per_class_f1": {},
    }

    profiles: List[EpochProfile] = []
    total_start = time.perf_counter()
    max_rss_after_forward_mb = 0.0
    max_rss_after_backward_mb = 0.0
    first_epoch_snapshots: List[Dict[str, float]] = []

    loader = GraphSAINTNodeSampler(
            data,
            batch_size=min(600, max(200, data.num_nodes // 2)),
            num_steps=5,
            sample_coverage=100,
            shuffle=True,
        )
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # loader = GraphSAINTNodeSampler(
        #     data,
        #     batch_size=min(600, max(200, data.num_nodes // 2)),
        #     num_steps=5,
        #     sample_coverage=100,
        #     shuffle=True,
        # )

        model.train()
        total_loss = 0.0
        batch_count = 0
        epoch_rss_after_forward_mb = 0.0
        epoch_rss_after_backward_mb = 0.0

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()

            snapshots, snapshot_fn = _make_snapshot_collector()
            use_snapshots = epoch == 1 and batch_idx == 0

            out = model(
                batch.x,
                batch.edge_index,
                snapshot_fn=snapshot_fn if use_snapshots else None,
            )

            if not hasattr(batch, "train_mask") or int(batch.train_mask.sum()) == 0:
                continue

            loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
            rss_after_forward_mb = _get_process_rss_mb()

            loss.backward()
            rss_after_backward_mb = _get_process_rss_mb()

            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

            epoch_rss_after_forward_mb = max(epoch_rss_after_forward_mb, rss_after_forward_mb)
            epoch_rss_after_backward_mb = max(epoch_rss_after_backward_mb, rss_after_backward_mb)

            if use_snapshots:
                first_epoch_snapshots = snapshots

        epoch_time_s = time.perf_counter() - epoch_start
        avg_loss = total_loss / max(batch_count, 1)
        metrics = _evaluate(model, data)

        if metrics["val_acc"] > best["val_acc"]:
            best = {
                "epoch": epoch,
                "val_acc": metrics["val_acc"],
                "test_acc": metrics["test_acc"],
                "test_macro_f1": metrics["test_macro_f1"],
                "per_class_f1": metrics["test_per_class_f1"],
            }

        max_rss_after_forward_mb = max(max_rss_after_forward_mb, epoch_rss_after_forward_mb)
        max_rss_after_backward_mb = max(max_rss_after_backward_mb, epoch_rss_after_backward_mb)

        profiles.append(
            EpochProfile(
                epoch=epoch,
                loss=float(avg_loss),
                epoch_time_s=float(epoch_time_s),
                rss_after_forward_mb=float(epoch_rss_after_forward_mb),
                rss_after_backward_mb=float(epoch_rss_after_backward_mb),
                forward_snapshots=first_epoch_snapshots if epoch == 1 else [],
            )
        )

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={avg_loss:.4f} | "
                f"val_acc={metrics['val_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | "
                f"test_macro_f1={metrics['test_macro_f1']:.4f} | "
                f"epoch_time={epoch_time_s:.4f}s"
            )

    total_time_s = time.perf_counter() - total_start
    avg_epoch_time_s = sum(p.epoch_time_s for p in profiles) / len(profiles)

    return {
        "best": best,
        "profiles": profiles,
        "total_time_s": total_time_s,
        "avg_epoch_time_s": avg_epoch_time_s,
        "max_rss_after_forward_mb": max_rss_after_forward_mb,
        "max_rss_after_backward_mb": max_rss_after_backward_mb,
    }


def train_clustergcn(
    model,
    data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
):
    try:
        cluster_data = ClusterData(data, num_parts=50, recursive=False)
        loader = ClusterLoader(cluster_data, batch_size=5, shuffle=True)
    except Exception as e:
        raise RuntimeError(
            f"ClusterGCN setup failed. This is usually a dependency/platform issue on Windows. Original error: {e}"
        ) from e

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {
        "epoch": -1,
        "val_acc": -1.0,
        "test_acc": 0.0,
        "test_macro_f1": 0.0,
        "per_class_f1": {},
    }

    profiles: List[EpochProfile] = []
    total_start = time.perf_counter()
    max_rss_after_forward_mb = 0.0
    max_rss_after_backward_mb = 0.0
    first_epoch_snapshots: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        batch_count = 0
        epoch_rss_after_forward_mb = 0.0
        epoch_rss_after_backward_mb = 0.0

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()

            snapshots, snapshot_fn = _make_snapshot_collector()
            use_snapshots = epoch == 1 and batch_idx == 0

            out = model(
                batch.x,
                batch.edge_index,
                snapshot_fn=snapshot_fn if use_snapshots else None,
            )

            if not hasattr(batch, "train_mask") or int(batch.train_mask.sum()) == 0:
                continue

            loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
            rss_after_forward_mb = _get_process_rss_mb()

            loss.backward()
            rss_after_backward_mb = _get_process_rss_mb()

            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

            epoch_rss_after_forward_mb = max(epoch_rss_after_forward_mb, rss_after_forward_mb)
            epoch_rss_after_backward_mb = max(epoch_rss_after_backward_mb, rss_after_backward_mb)

            if use_snapshots:
                first_epoch_snapshots = snapshots

        epoch_time_s = time.perf_counter() - epoch_start
        avg_loss = total_loss / max(batch_count, 1)
        metrics = _evaluate(model, data)

        if metrics["val_acc"] > best["val_acc"]:
            best = {
                "epoch": epoch,
                "val_acc": metrics["val_acc"],
                "test_acc": metrics["test_acc"],
                "test_macro_f1": metrics["test_macro_f1"],
                "per_class_f1": metrics["test_per_class_f1"],
            }

        max_rss_after_forward_mb = max(max_rss_after_forward_mb, epoch_rss_after_forward_mb)
        max_rss_after_backward_mb = max(max_rss_after_backward_mb, epoch_rss_after_backward_mb)

        profiles.append(
            EpochProfile(
                epoch=epoch,
                loss=float(avg_loss),
                epoch_time_s=float(epoch_time_s),
                rss_after_forward_mb=float(epoch_rss_after_forward_mb),
                rss_after_backward_mb=float(epoch_rss_after_backward_mb),
                forward_snapshots=first_epoch_snapshots if epoch == 1 else [],
            )
        )

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={avg_loss:.4f} | "
                f"val_acc={metrics['val_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | "
                f"test_macro_f1={metrics['test_macro_f1']:.4f} | "
                f"epoch_time={epoch_time_s:.4f}s"
            )

    total_time_s = time.perf_counter() - total_start
    avg_epoch_time_s = sum(p.epoch_time_s for p in profiles) / len(profiles)

    return {
        "best": best,
        "profiles": profiles,
        "total_time_s": total_time_s,
        "avg_epoch_time_s": avg_epoch_time_s,
        "max_rss_after_forward_mb": max_rss_after_forward_mb,
        "max_rss_after_backward_mb": max_rss_after_backward_mb,
    }