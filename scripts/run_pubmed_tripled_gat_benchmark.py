from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_pubmed_tripled import load_pubmed_tripled, sample_pubmed_subgraph
from src.gat_model import SimpleGATNet
from src.gat_trainer import (
    train_gat_computational_memory,
    train_gat_rss,
)


def parse_args():
    p = argparse.ArgumentParser()
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


def _print_rss_tables(all_rows):
    print("\n" + "=" * 104)
    print("PUBMED (TRIPLED FEATURES) — GAT RSS RESULTS")
    print("=" * 104)
    print(f"{'Fraction':<10}{'Nodes':<10}{'Forward Peak (MB)':<20}{'Overhead (MB)':<18}{'Time (ms)':<12}{'Test Acc':<10}")
    print("-" * 104)
    for r in all_rows:
        print(
            f"{int(r['fraction']*100):<10}"
            f"{r['nodes']:<10}"
            f"{r['forward_peak_mb']:<20.2f}"
            f"{r['forward_overhead_mb']:<18.2f}"
            f"{r['forward_time_ms']:<12.2f}"
            f"{r['best_test_acc']:<10.4f}"
        )
    print()


def _print_comp_tables(all_rows):
    print("\n" + "=" * 118)
    print("PUBMED (TRIPLED FEATURES) — GAT COMPUTATIONAL MEMORY RESULTS")
    print("=" * 118)
    print(
        f"{'Fraction':<10}{'Nodes':<10}{'Forward MB':<14}{'Backward MB':<14}"
        f"{'Step MB':<12}{'Attention MB':<15}{'Epoch Time (s)':<16}{'Test Acc':<10}"
    )
    print("-" * 118)
    for r in all_rows:
        print(
            f"{int(r['fraction']*100):<10}"
            f"{r['nodes']:<10}"
            f"{r['forward_peak_mb']:<14.2f}"
            f"{r['backward_peak_mb']:<14.2f}"
            f"{r['step_peak_mb']:<12.2f}"
            f"{r['attention_mb']:<15.2f}"
            f"{r['avg_epoch_time_s']:<16.4f}"
            f"{r['best_test_acc']:<10.4f}"
        )
    print()


def main():
    args = parse_args()
    dataset, full_data = load_pubmed_tripled(root=args.root, seed=args.seed)

    fractions = [0.25, 0.5, 0.75, 1.0]
    all_rows = []

    print("Loaded PubMed with tripled features for GAT")
    print(f"Nodes: {full_data.num_nodes}")
    print(f"Edges: {full_data.num_edges}")
    print(f"Input features: {full_data.num_node_features}")
    print(f"Classes: {dataset.num_classes}")

    for fraction in fractions:
        print("\n" + "=" * 78)
        print(f"Running gat | fraction={fraction} | dataset=pubmed_tripled | memory_mode={args.memory_mode}")
        print("=" * 78)

        data = sample_pubmed_subgraph(full_data, fraction=fraction, seed=args.seed)
        model = SimpleGATNet(
            data.num_node_features,
            args.hidden_dim,
            dataset.num_classes,
            heads=args.heads,
            dropout=args.dropout,
        )

        if args.memory_mode == "computational":
            summary = train_gat_computational_memory(
                model,
                data,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            cm = summary["computational_memory"]
            all_rows.append({
                "fraction": fraction,
                "nodes": int(data.num_nodes),
                "edges": int(data.num_edges),
                "best_test_acc": summary["best_test_acc"],
                "avg_epoch_time_s": summary["avg_epoch_time"],
                "forward_peak_mb": cm["forward_peak_mb_mean"],
                "backward_peak_mb": cm["backward_peak_mb_mean"],
                "step_peak_mb": cm["step_peak_mb_mean"],
                "attention_mb": cm["attention_mb_mean"],
            })
        else:
            summary = train_gat_rss(
                model,
                data,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            fwd = summary.ram_stats.get("forward", {})
            all_rows.append({
                "fraction": fraction,
                "nodes": int(data.num_nodes),
                "edges": int(data.num_edges),
                "best_test_acc": summary.best_test_acc,
                "forward_peak_mb": fwd.get("peak_mb_mean", 0.0),
                "forward_overhead_mb": fwd.get("overhead_mb_mean", 0.0),
                "forward_time_ms": fwd.get("duration_s_mean", 0.0) * 1000,
            })

    if args.memory_mode == "rss":
        _print_rss_tables(all_rows)
    else:
        _print_comp_tables(all_rows)


if __name__ == "__main__":
    main()