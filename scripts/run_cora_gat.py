from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import load_cora, sample_cora_subgraph
from src.gat_model import SimpleGATNet
from src.gat_trainer import (
    train_gat_computational_memory,
    train_gat_rss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root", type=str, default=os.path.join(ROOT, "data"))
    p.add_argument("--memory-mode", type=str, default="computational", choices=["rss", "computational"])
    return p.parse_args()


def main():
    args = parse_args()
    dataset, full_data = load_cora(root=args.root)
    data = sample_cora_subgraph(full_data, fraction=args.fraction, seed=args.seed)

    print("Loaded Cora for model=gat")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Input features: {data.num_node_features}")
    print(f"Classes: {dataset.num_classes}")
    print(
        f"Split sizes: train={int(data.train_mask.sum())}, "
        f"val={int(data.val_mask.sum())}, "
        f"test={int(data.test_mask.sum())}"
    )

    model = SimpleGATNet(
        data.num_node_features,
        args.hidden_dim,
        dataset.num_classes,
        heads=args.heads,
        dropout=args.dropout,
    )

    if args.memory_mode == "computational":
        results = train_gat_computational_memory(
            model,
            data,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        print("\nBest results based on validation accuracy")
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
        results = train_gat_rss(
            model,
            data,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        print("\nBest results based on validation accuracy")
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