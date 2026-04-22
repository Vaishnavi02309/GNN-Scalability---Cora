from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import build_seed_growth_order, load_cora, subgraph_from_growth_order
from src.models import ClusterGCNNet, GraphSAGENet
from src.trainers import (
    train_clustergcn,
    train_clustergcn_computational_memory,
    train_full_batch,
    train_graphsage_computational_memory,
    train_graphsaint,
    train_graphsaint_computational_memory,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="graphsage", choices=["graphsage", "graphsaint", "clustergcn"])
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--memory-mode", type=str, default="rss", choices=["rss", "computational"])
    return p.parse_args()


def main():
    args = parse_args()
    dataset, full_data = load_cora(args.root)

    growth_order, seed_nodes = build_seed_growth_order(
        full_data,
        num_seed_nodes=6,
        seed=args.seed,
    )

    data = subgraph_from_growth_order(
        full_data,
        growth_order=growth_order,
        fraction=args.fraction,
    )

    print(f"Seed nodes: {seed_nodes.tolist()}")
    print(f"Loaded Cora for model={args.model}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Input features: {data.num_node_features}")
    print(f"Classes: {dataset.num_classes}")
    print(
        f"Split sizes: train={int(data.train_mask.sum())}, "
        f"val={int(data.val_mask.sum())}, "
        f"test={int(data.test_mask.sum())}"
    )

    if args.model == "graphsage":
        model = GraphSAGENet(
            data.num_node_features,
            args.hidden_dim,
            dataset.num_classes,
            dropout=args.dropout,
        )

        if args.memory_mode == "computational":
            results = train_graphsage_computational_memory(
                model,
                data,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            results = train_full_batch(
                model,
                data,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

    elif args.model == "graphsaint":
        model = GraphSAGENet(
            data.num_node_features,
            args.hidden_dim,
            dataset.num_classes,
            dropout=args.dropout,
        )

        if args.memory_mode == "computational":
            results = train_graphsaint_computational_memory(
                model,
                data,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            results = train_graphsaint(
                model,
                data,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

    elif args.model == "clustergcn":
        model = ClusterGCNNet(
            data.num_node_features,
            args.hidden_dim,
            dataset.num_classes,
            dropout=args.dropout,
        )

        if args.memory_mode == "computational":
            results = train_clustergcn_computational_memory(
                model,
                data,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            results = train_clustergcn(
                model,
                data,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    print("\nBest results based on validation accuracy")

    if args.memory_mode == "computational":
        print(f"Best epoch: {results['best_epoch']}")
        print(f"Best val accuracy: {results['best_val_acc']:.4f}")
        print(f"Best test accuracy: {results['best_test_acc']:.4f}")
        print(f"Best test macro F1: {results['best_test_macro_f1']:.4f}")
        print(f"Per-class F1: {results['best_per_class_f1']}")
        print(f"Total training time: {results['total_training_time']:.2f}s")
        print(f"Average epoch time: {results['avg_epoch_time']:.4f}s")

        print("\nComputational memory:")
        for k, v in results["computational_memory"].items():
            print(f"  {k}: {v:.4f}")
    else:
        print(f"Best epoch: {results.best_epoch}")
        print(f"Best val accuracy: {results.best_val_acc:.4f}")
        print(f"Best test accuracy: {results.best_test_acc:.4f}")
        print(f"Best test macro F1: {results.best_test_macro_f1:.4f}")
        print(f"Per-class F1: {results.best_per_class_f1}")
        print(f"Total training time: {results.total_training_time:.2f}s")
        print(f"Average epoch time: {results.avg_epoch_time:.4f}s")
        print("RAM stats:")
        print(results.ram_stats)


if __name__ == "__main__":
    main()