from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from src.profiling import EpochProfiler


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def tensor_nbytes(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()


def module_param_nbytes(module: torch.nn.Module) -> int:
    return sum(p.element_size() * p.nelement() for p in module.parameters())


def module_grad_nbytes(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.element_size() * p.nelement()
    return total


def optimizer_state_nbytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.element_size() * v.nelement()
    return total


def bytes_to_mb(nbytes: int) -> float:
    return nbytes / (1024 * 1024)


def _full_data_tensor_memory_mb(data) -> float:
    total = 0
    for name in ["x", "edge_index", "y", "train_mask", "val_mask", "test_mask"]:
        if hasattr(data, name):
            obj = getattr(data, name)
            if torch.is_tensor(obj):
                total += tensor_nbytes(obj)
    return bytes_to_mb(total)


def _activation_memory_mb(activations: dict[str, torch.Tensor]) -> float:
    total = 0
    for t in activations.values():
        if torch.is_tensor(t):
            total += tensor_nbytes(t)
    return bytes_to_mb(total)


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


# -----------------------------------------------------------------------------
# RSS summary dataclass
# -----------------------------------------------------------------------------

@dataclass
class RunSummary:
    best_epoch: int
    best_val_acc: float
    best_test_acc: float
    best_test_macro_f1: float
    best_per_class_f1: Dict[int, float]
    avg_epoch_time: float
    total_training_time: float
    ram_stats: Dict[str, Dict[str, float]]


# -----------------------------------------------------------------------------
# RSS-based trainer
# -----------------------------------------------------------------------------

def train_gat_rss(
    model,
    data,
    epochs: int = 200,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    warmup_epochs: int = 5,
) -> RunSummary:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    profiler = EpochProfiler()

    best_epoch, best_val_acc = -1, -1.0
    best_test_acc, best_test_f1 = 0.0, 0.0
    best_per_class_f1 = {}

    epoch_times = []
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        optimizer.zero_grad()

        profiler.begin_epoch(epoch)

        with profiler.phase("forward", force_gc=False):
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        with profiler.phase("backward", force_gc=False):
            loss.backward()

        with profiler.phase("step", force_gc=False):
            optimizer.step()

        profiler.end_epoch()

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        metrics = _evaluate(model, data)

        if metrics["val_acc"] > best_val_acc:
            best_epoch = epoch
            best_val_acc = metrics["val_acc"]
            best_test_acc = metrics["test_acc"]
            best_test_f1 = metrics["test_macro_f1"]
            best_per_class_f1 = metrics["test_per_class_f1"]

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_acc={metrics['val_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | "
                f"test_macro_f1={metrics['test_macro_f1']:.4f} | "
                f"epoch_time={epoch_time:.4f}s"
            )

    total_time = time.perf_counter() - total_start
    profiler.print_summary(warmup_epochs=warmup_epochs)

    return RunSummary(
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_test_acc=best_test_acc,
        best_test_macro_f1=best_test_f1,
        best_per_class_f1=best_per_class_f1,
        avg_epoch_time=sum(epoch_times) / len(epoch_times),
        total_training_time=total_time,
        ram_stats=profiler.slim_summary(warmup_epochs=warmup_epochs),
    )


# -----------------------------------------------------------------------------
# Computational-memory trainer
# -----------------------------------------------------------------------------

def train_gat_computational_memory(
    model,
    data,
    epochs: int = 20,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_epoch, best_val_acc = -1, -1.0
    best_test_acc, best_test_f1 = 0.0, 0.0
    best_per_class_f1 = {}

    epoch_times = []
    total_start = time.perf_counter()
    memory_rows = []

    input_mb = _full_data_tensor_memory_mb(data)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        out, activations = model(data.x, data.edge_index, return_activations=True)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        param_mb = bytes_to_mb(module_param_nbytes(model))
        act_mb = _activation_memory_mb(activations)
        loss_mb = bytes_to_mb(tensor_nbytes(loss))

        # Attention-only memory = alpha tensors from both layers
        attention_mb = bytes_to_mb(
            tensor_nbytes(activations["attn_alpha1"]) +
            tensor_nbytes(activations["attn_alpha2"])
        )

        forward_comp_mb = param_mb + input_mb + act_mb + loss_mb

        loss.backward()
        grad_mb = bytes_to_mb(module_grad_nbytes(model))
        backward_comp_mb = param_mb + input_mb + act_mb + grad_mb

        optimizer.step()
        opt_state_mb = bytes_to_mb(optimizer_state_nbytes(optimizer))
        step_comp_mb = param_mb + grad_mb + opt_state_mb

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        metrics = _evaluate(model, data)

        if metrics["val_acc"] > best_val_acc:
            best_epoch = epoch
            best_val_acc = metrics["val_acc"]
            best_test_acc = metrics["test_acc"]
            best_test_f1 = metrics["test_macro_f1"]
            best_per_class_f1 = metrics["test_per_class_f1"]

        memory_rows.append({
            "epoch": epoch,
            "forward_comp_peak_mb": forward_comp_mb,
            "backward_comp_peak_mb": backward_comp_mb,
            "step_comp_peak_mb": step_comp_mb,
            "attention_mb": attention_mb,
        })

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_acc={metrics['val_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | "
                f"test_macro_f1={metrics['test_macro_f1']:.4f} | "
                f"epoch_time={epoch_time:.4f}s | "
                f"forward_comp_peak={forward_comp_mb:.2f} MB | "
                f"attention_memory={attention_mb:.2f} MB"
            )

    total_time = time.perf_counter() - total_start

    warmup = 5
    usable = memory_rows[warmup:] if len(memory_rows) > warmup else memory_rows

    def mean_std(key: str):
        vals = [r[key] for r in usable]
        mean = sum(vals) / len(vals) if vals else 0.0
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = var ** 0.5
        else:
            std = 0.0
        return mean, std

    f_mean, f_std = mean_std("forward_comp_peak_mb")
    b_mean, b_std = mean_std("backward_comp_peak_mb")
    s_mean, s_std = mean_std("step_comp_peak_mb")
    a_mean, a_std = mean_std("attention_mb")

    print()
    print("=" * 72)
    print("GAT COMPUTATIONAL MEMORY SUMMARY")
    print("=" * 72)
    print(f"Forward peak:    {f_mean:.2f} ± {f_std:.2f} MB")
    print(f"Backward peak:   {b_mean:.2f} ± {b_std:.2f} MB")
    print(f"Step peak:       {s_mean:.2f} ± {s_std:.2f} MB")
    print(f"Attention only:  {a_mean:.2f} ± {a_std:.2f} MB")
    print("=" * 72)
    print()

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_f1,
        "best_per_class_f1": best_per_class_f1,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        "total_training_time": total_time,
        "computational_memory": {
            "forward_peak_mb_mean": f_mean,
            "forward_peak_mb_std": f_std,
            "backward_peak_mb_mean": b_mean,
            "backward_peak_mb_std": b_std,
            "step_peak_mb_mean": s_mean,
            "step_peak_mb_std": s_std,
            "attention_mb_mean": a_mean,
            "attention_mb_std": a_std,
        },
    }