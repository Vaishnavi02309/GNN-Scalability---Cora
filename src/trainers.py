from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import ClusterData, ClusterLoader, GraphSAINTNodeSampler

from .profiling import EpochProfiler


@dataclass
class RunSummary:
    best_epoch:          int
    best_val_acc:        float
    best_test_acc:       float
    best_test_macro_f1:  float
    best_per_class_f1:   Dict[int, float]
    avg_epoch_time:      float
    total_training_time: float
    ram_stats:           Dict[str, Dict[str, float]]


def _safe_macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _safe_per_class_f1(y_true, y_pred) -> Dict[int, float]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    scores = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {int(l): float(s) for l, s in zip(labels, scores)}


@torch.no_grad()
def _evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    metrics = {}
    for split, mask_name in [("train", "train_mask"), ("val", "val_mask"), ("test", "test_mask")]:
        mask = getattr(data, mask_name)
        y_true = data.y[mask].cpu()
        y_pred = pred[mask].cpu()
        if y_true.numel() == 0:
            metrics[f"{split}_acc"] = 0.0
            metrics[f"{split}_macro_f1"] = 0.0
            metrics[f"{split}_per_class_f1"] = {}
            continue
        metrics[f"{split}_acc"] = float(accuracy_score(y_true, y_pred))
        metrics[f"{split}_macro_f1"] = float(_safe_macro_f1(y_true, y_pred))
        metrics[f"{split}_per_class_f1"] = _safe_per_class_f1(y_true, y_pred)
    return metrics


def train_full_batch(
    model,
    data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    warmup_epochs: int = 5,
) -> RunSummary:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    profiler  = EpochProfiler()

    best_epoch, best_val_acc    = -1, -1.0
    best_test_acc, best_test_f1 = 0.0, 0.0
    best_per_class_f1           = {}
    epoch_times = []
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        profiler.begin_epoch(epoch)

        with profiler.phase("forward"):
            out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        with profiler.phase("backward"):
            loss.backward()

        with profiler.phase("step"):
            optimizer.step()

        profiler.end_epoch()

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        metrics = _evaluate(model, data)

        if metrics["val_acc"] > best_val_acc:
            best_epoch        = epoch
            best_val_acc      = metrics["val_acc"]
            best_test_acc     = metrics["test_acc"]
            best_test_f1      = metrics["test_macro_f1"]
            best_per_class_f1 = metrics["test_per_class_f1"]

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_acc={metrics['val_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | test_macro_f1={metrics['test_macro_f1']:.4f} | "
                f"epoch_time={epoch_time:.4f}s"
            )

    total_time = time.perf_counter() - total_start
    profiler.print_summary(warmup_epochs=warmup_epochs)

    return RunSummary(
        best_epoch          = best_epoch,
        best_val_acc        = best_val_acc,
        best_test_acc       = best_test_acc,
        best_test_macro_f1  = best_test_f1,
        best_per_class_f1   = best_per_class_f1,
        avg_epoch_time      = sum(epoch_times) / len(epoch_times),
        total_training_time = total_time,
        ram_stats           = profiler.slim_summary(warmup_epochs=warmup_epochs),
    )


def train_graphsaint(
    model,
    data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    warmup_epochs: int = 5,
) -> RunSummary:
    loader = GraphSAINTNodeSampler(
        data,
        batch_size=min(600, max(200, data.num_nodes // 2)),
        num_steps=5,
        sample_coverage=100,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    profiler  = EpochProfiler()

    best_epoch, best_val_acc    = -1, -1.0
    best_test_acc, best_test_f1 = 0.0, 0.0
    best_per_class_f1           = {}
    epoch_times = []
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, batch_count = 0.0, 0

        profiler.begin_epoch(epoch)

        for batch in loader:
            optimizer.zero_grad()

            if not hasattr(batch, "train_mask") or int(batch.train_mask.sum()) == 0:
                continue

            with profiler.phase("forward"):
                out = model(batch.x, batch.edge_index)

            loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])

            with profiler.phase("backward"):
                loss.backward()

            with profiler.phase("step"):
                optimizer.step()

            total_loss  += float(loss.item())
            batch_count += 1

        profiler.end_epoch()

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = total_loss / max(batch_count, 1)
        metrics  = _evaluate(model, data)

        if metrics["val_acc"] > best_val_acc:
            best_epoch        = epoch
            best_val_acc      = metrics["val_acc"]
            best_test_acc     = metrics["test_acc"]
            best_test_f1      = metrics["test_macro_f1"]
            best_per_class_f1 = metrics["test_per_class_f1"]

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | loss={avg_loss:.4f} | val_acc={metrics['val_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | test_macro_f1={metrics['test_macro_f1']:.4f} | "
                f"epoch_time={epoch_time:.4f}s"
            )

    total_time = time.perf_counter() - total_start
    profiler.print_summary(warmup_epochs=warmup_epochs)

    return RunSummary(
        best_epoch          = best_epoch,
        best_val_acc        = best_val_acc,
        best_test_acc       = best_test_acc,
        best_test_macro_f1  = best_test_f1,
        best_per_class_f1   = best_per_class_f1,
        avg_epoch_time      = sum(epoch_times) / len(epoch_times),
        total_training_time = total_time,
        ram_stats           = profiler.slim_summary(warmup_epochs=warmup_epochs),
    )


def train_clustergcn(
    model,
    data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    warmup_epochs: int = 5,
) -> RunSummary:
    try:
        cluster_data = ClusterData(data, num_parts=50, recursive=False)
        loader       = ClusterLoader(cluster_data, batch_size=5, shuffle=True)
    except Exception as e:
        raise RuntimeError(
            f"ClusterGCN setup failed. This is usually a dependency/platform issue on Windows. Original error: {e}"
        ) from e

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    profiler  = EpochProfiler()

    best_epoch, best_val_acc    = -1, -1.0
    best_test_acc, best_test_f1 = 0.0, 0.0
    best_per_class_f1           = {}
    epoch_times = []
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, batch_count = 0.0, 0

        profiler.begin_epoch(epoch)

        for batch in loader:
            optimizer.zero_grad()

            if not hasattr(batch, "train_mask") or int(batch.train_mask.sum()) == 0:
                continue

            with profiler.phase("forward"):
                out = model(batch.x, batch.edge_index)

            loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])

            with profiler.phase("backward"):
                loss.backward()

            with profiler.phase("step"):
                optimizer.step()

            total_loss  += float(loss.item())
            batch_count += 1

        profiler.end_epoch()

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = total_loss / max(batch_count, 1)
        metrics  = _evaluate(model, data)

        if metrics["val_acc"] > best_val_acc:
            best_epoch        = epoch
            best_val_acc      = metrics["val_acc"]
            best_test_acc     = metrics["test_acc"]
            best_test_f1      = metrics["test_macro_f1"]
            best_per_class_f1 = metrics["test_per_class_f1"]

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | loss={avg_loss:.4f} | val_acc={metrics['val_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | test_macro_f1={metrics['test_macro_f1']:.4f} | "
                f"epoch_time={epoch_time:.4f}s"
            )

    total_time = time.perf_counter() - total_start
    profiler.print_summary(warmup_epochs=warmup_epochs)

    return RunSummary(
        best_epoch          = best_epoch,
        best_val_acc        = best_val_acc,
        best_test_acc       = best_test_acc,
        best_test_macro_f1  = best_test_f1,
        best_per_class_f1   = best_per_class_f1,
        avg_epoch_time      = sum(epoch_times) / len(epoch_times),
        total_training_time = total_time,
        ram_stats           = profiler.slim_summary(warmup_epochs=warmup_epochs),
    )