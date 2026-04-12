from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import load_cora, sample_cora_subgraph
from src.models import build_model
from src.trainers import train_full_batch


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
    return p.parse_args()


def main():
    args = parse_args()
    dataset, data = load_cora(root=os.path.join(ROOT, "data"))
    data = sample_cora_subgraph(data, fraction=args.fraction, seed=args.seed)

    print(f"Loaded Cora for model={args.model}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Input features: {data.num_node_features}")
    print(f"Classes: {dataset.num_classes}")
    print(
        f"Split sizes: train={int(data.train_mask.sum())}, val={int(data.val_mask.sum())}, test={int(data.test_mask.sum())}"
    )

    model = build_model(args.model, data.num_node_features, args.hidden_dim, dataset.num_classes, args.dropout)
    results = train_full_batch(model, data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)

    best = results["best"]
    print("\nBest results based on validation accuracy")
    print(f"Best epoch: {best['epoch']}")
    print(f"Best val accuracy: {best['val_acc']:.4f}")
    print(f"Best test accuracy: {best['test_acc']:.4f}")
    print(f"Best test macro F1: {best['test_macro_f1']:.4f}")
    print(f"Per-class F1: {best['per_class_f1']}")
    print(f"Total training time: {results['total_time_s']:.2f}s")
    print(f"Average epoch time: {results['avg_epoch_time_s']:.4f}s")
    print(f"Max RSS after forward: {results['max_rss_after_forward_mb']:.2f} MB")
    print(f"Max RSS after backward: {results['max_rss_after_backward_mb']:.2f} MB")

    first = results['profiles'][0]
    print("\nForward-pass RAM snapshots from epoch 1")
    for snap in first.forward_snapshots:
        print(f"  {snap['step']}: {snap['rss_mb']:.2f} MB")


if __name__ == "__main__":
    main()
