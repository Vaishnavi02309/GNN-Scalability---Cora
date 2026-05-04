from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import ClusterData, ClusterLoader, GraphSAINTNodeSampler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_pubmed_tripled import load_pubmed_tripled, sample_pubmed_subgraph
from src.models import build_model
from src.gat_model import SimpleGATNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_probs(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)
    pred = probs.argmax(dim=1)

    test_mask = data.test_mask
    y_test = data.y[test_mask]
    pred_test = pred[test_mask]
    probs_test = probs[test_mask]

    acc = float((pred_test == y_test).float().mean().item())
    return acc, probs_test.cpu(), y_test.cpu()


def train_full_batch_return_probs(model, data, epochs, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_acc = -1.0
    epoch_times = []

    for _ in range(1, epochs + 1):
        start = time.perf_counter()

        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        epoch_times.append(time.perf_counter() - start)

        with torch.no_grad():
            model.eval()
            out_eval = model(data.x, data.edge_index)
            pred = out_eval.argmax(dim=1)
            val_acc = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean().item())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, probs_test, y_test = evaluate_probs(model, data)

    return {
        "test_acc": test_acc,
        "probs_test": probs_test,
        "y_test": y_test,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        "best_val_acc": best_val_acc,
    }


def train_graphsaint_return_probs(model, data, epochs, lr, weight_decay):
    loader = GraphSAINTNodeSampler(
        data,
        batch_size=min(600, max(200, data.num_nodes // 2)),
        num_steps=5,
        sample_coverage=100,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_acc = -1.0
    epoch_times = []

    for _ in range(1, epochs + 1):
        start = time.perf_counter()

        model.train()

        for batch in loader:
            if not hasattr(batch, "train_mask") or int(batch.train_mask.sum()) == 0:
                continue

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

        epoch_times.append(time.perf_counter() - start)

        with torch.no_grad():
            model.eval()
            out_eval = model(data.x, data.edge_index)
            pred = out_eval.argmax(dim=1)
            val_acc = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean().item())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, probs_test, y_test = evaluate_probs(model, data)

    return {
        "test_acc": test_acc,
        "probs_test": probs_test,
        "y_test": y_test,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        "best_val_acc": best_val_acc,
    }


def train_clustergcn_return_probs(
    model,
    data,
    epochs,
    lr,
    weight_decay,
    cluster_data,
):
    assert cluster_data is not None, "ClusterData must be created once in main() and passed here."

    loader = ClusterLoader(cluster_data, batch_size=5, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_acc = -1.0
    epoch_times = []

    for _ in range(1, epochs + 1):
        start = time.perf_counter()

        model.train()

        for batch in loader:
            if not hasattr(batch, "train_mask") or int(batch.train_mask.sum()) == 0:
                continue

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

        epoch_times.append(time.perf_counter() - start)

        with torch.no_grad():
            model.eval()
            out_eval = model(data.x, data.edge_index)
            pred = out_eval.argmax(dim=1)
            val_acc = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean().item())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, probs_test, y_test = evaluate_probs(model, data)

    return {
        "test_acc": test_acc,
        "probs_test": probs_test,
        "y_test": y_test,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        "best_val_acc": best_val_acc,
    }


def compute_bias_variance(seed_outputs: List[Dict], num_classes: int):
    probs_all = torch.stack([r["probs_test"] for r in seed_outputs], dim=0)
    y_test = seed_outputs[0]["y_test"]

    mean_probs = probs_all.mean(dim=0)
    true_onehot = F.one_hot(y_test, num_classes=num_classes).float()

    bias_sq = ((mean_probs - true_onehot) ** 2).sum(dim=1).mean().item()
    variance = ((probs_all - mean_probs.unsqueeze(0)) ** 2).sum(dim=2).mean().item()

    accs = [r["test_acc"] for r in seed_outputs]
    times = [r["avg_epoch_time"] for r in seed_outputs]

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "time_mean": float(np.mean(times)),
        "time_std": float(np.std(times)),
        "bias_sq": bias_sq,
        "variance": variance,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--root", type=str, default=os.path.join(ROOT, "data"))
    p.add_argument("--graph-seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["graphsage", "graphsaint", "clustergcn", "gat"],
        choices=["graphsage", "graphsaint", "clustergcn", "gat"],
    )
    return p.parse_args()


def main():
    args = parse_args()
    dataset, full_data = load_pubmed_tripled(root=args.root, seed=args.graph_seed)

    rows = []

    print("Loaded PubMed with tripled features")
    print(f"Nodes: {full_data.num_nodes}")
    print(f"Edges: {full_data.num_edges}")
    print(f"Input features: {full_data.num_node_features}")
    print(f"Classes: {dataset.num_classes}")

    for fraction in args.fractions:
        data = sample_pubmed_subgraph(full_data, fraction=fraction, seed=args.graph_seed)

        print("\n" + "=" * 88)
        print(f"PubMed-tripled fraction={fraction} | nodes={data.num_nodes} | edges={data.num_edges}")
        print("=" * 88)

        for model_name in args.models:
            print(f"\nRunning model={model_name} over seeds={args.seeds}")

            cluster_data = None
            if model_name == "clustergcn":
                print("Computing METIS partitioning once for all seeds...")
                cluster_data = ClusterData(data, num_parts=50, recursive=False)
                print("Done!")

            seed_outputs = []

            for seed in args.seeds:
                set_seed(seed)

                if model_name == "gat":
                    model = SimpleGATNet(
                        data.num_node_features,
                        hidden_channels=8,
                        out_channels=dataset.num_classes,
                        heads=8,
                        dropout=0.6,
                    )
                    result = train_full_batch_return_probs(
                        model,
                        data,
                        epochs=args.epochs,
                        lr=0.005,
                        weight_decay=args.weight_decay,
                    )

                else:
                    model = build_model(
                        model_name,
                        data.num_node_features,
                        args.hidden_dim,
                        dataset.num_classes,
                        args.dropout,
                    )

                    if model_name == "graphsage":
                        result = train_full_batch_return_probs(
                            model,
                            data,
                            args.epochs,
                            args.lr,
                            args.weight_decay,
                        )
                    elif model_name == "graphsaint":
                        result = train_graphsaint_return_probs(
                            model,
                            data,
                            args.epochs,
                            args.lr,
                            args.weight_decay,
                        )
                    elif model_name == "clustergcn":
                        result = train_clustergcn_return_probs(
                            model,
                            data,
                            args.epochs,
                            args.lr,
                            args.weight_decay,
                            cluster_data=cluster_data,
                        )
                    else:
                        raise ValueError(model_name)

                seed_outputs.append(result)

                print(
                    f"  seed={seed} | test_acc={result['test_acc']:.4f} | "
                    f"best_val={result['best_val_acc']:.4f} | "
                    f"avg_epoch_time={result['avg_epoch_time']:.4f}s"
                )

            stats = compute_bias_variance(seed_outputs, dataset.num_classes)

            rows.append(
                {
                    "model": model_name,
                    "fraction": fraction,
                    "nodes": int(data.num_nodes),
                    "edges": int(data.num_edges),
                    **stats,
                }
            )

    print("\n" + "=" * 120)
    print("PUBMED-TRIPLED MULTI-SEED BIAS-VARIANCE SUMMARY")
    print("=" * 120)
    print(
        f"{'Model':<14}{'Frac':<8}{'Nodes':<8}"
        f"{'Acc Mean':<12}{'Acc Std':<12}"
        f"{'Bias^2':<12}{'Variance':<12}"
        f"{'Epoch Time':<14}"
    )
    print("-" * 120)

    for r in rows:
        print(
            f"{r['model']:<14}"
            f"{r['fraction']:<8.2f}"
            f"{r['nodes']:<8}"
            f"{r['acc_mean']:<12.4f}"
            f"{r['acc_std']:<12.4f}"
            f"{r['bias_sq']:<12.6f}"
            f"{r['variance']:<12.6f}"
            f"{r['time_mean']:<14.4f}"
        )


if __name__ == "__main__":
    main()