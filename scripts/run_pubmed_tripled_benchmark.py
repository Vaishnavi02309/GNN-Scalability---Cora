from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_pubmed_tripled import load_pubmed_tripled, sample_pubmed_subgraph
from src.models import build_model
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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root", type=str, default=os.path.join(ROOT, "data"))
    p.add_argument("--memory-mode", type=str, default="computational", choices=["rss", "computational"])
    return p.parse_args()


def _train(model_name, model, data, epochs, memory_mode):
    if memory_mode == "rss":
        if model_name == "graphsage":
            return train_full_batch(model, data, epochs=epochs)
        elif model_name == "graphsaint":
            return train_graphsaint(model, data, epochs=epochs)
        elif model_name == "clustergcn":
            return train_clustergcn(model, data, epochs=epochs)
    else:
        if model_name == "graphsage":
            return train_graphsage_computational_memory(model, data, epochs=epochs)
        elif model_name == "graphsaint":
            return train_graphsaint_computational_memory(model, data, epochs=epochs)
        elif model_name == "clustergcn":
            return train_clustergcn_computational_memory(model, data, epochs=epochs)

    raise ValueError(f"Unknown model: {model_name}")


def _print_rss_tables(all_rows):
    models = sorted(set(r["model"] for r in all_rows))

    print("\n" + "=" * 92)
    print("PUBMED (TRIPLED FEATURES) — FORWARD PROCESS MEMORY / RSS")
    print("=" * 92)
    header = f"  {'Model':<12}" + "".join(
        f"  {int(r['fraction']*100)}% ({r['nodes']}n)"
        for r in all_rows if r["model"] == models[0]
    )
    print(header)
    print("-" * 92)
    for m in models:
        row_vals = [r for r in all_rows if r["model"] == m]
        print(f"  {m:<12}" + "".join(f"  {r['forward_peak_mb']:>10.1f} MB" for r in row_vals))

    print("\n" + "=" * 92)
    print("PUBMED (TRIPLED FEATURES) — FORWARD RSS OVERHEAD (MB)")
    print("=" * 92)
    print(header)
    print("-" * 92)
    for m in models:
        row_vals = [r for r in all_rows if r["model"] == m]
        print(f"  {m:<12}" + "".join(f"  {r['forward_overhead_mb']:>10.1f} MB" for r in row_vals))

    print("\n" + "=" * 92)
    print("PUBMED (TRIPLED FEATURES) — FORWARD TIME (ms)")
    print("=" * 92)
    print(header)
    print("-" * 92)
    for m in models:
        row_vals = [r for r in all_rows if r["model"] == m]
        print(f"  {m:<12}" + "".join(f"  {r['forward_time_ms']:>10.1f} ms" for r in row_vals))

    print("\n" + "=" * 92)
    print("PUBMED (TRIPLED FEATURES) — BEST TEST ACCURACY")
    print("=" * 92)
    print(header)
    print("-" * 92)
    for m in models:
        row_vals = [r for r in all_rows if r["model"] == m]
        print(f"  {m:<12}" + "".join(f"  {r['best_test_acc']:>11.4f}" for r in row_vals))
    print()


def _print_comp_tables(all_rows):
    models = sorted(set(r["model"] for r in all_rows))

    print("\n" + "=" * 92)
    print("PUBMED (TRIPLED FEATURES) — FORWARD COMPUTATIONAL MEMORY (MB)")
    print("=" * 92)
    header = f"  {'Model':<12}" + "".join(
        f"  {int(r['fraction']*100)}% ({r['nodes']}n)"
        for r in all_rows if r["model"] == models[0]
    )
    print(header)
    print("-" * 92)
    for m in models:
        row_vals = [r for r in all_rows if r["model"] == m]
        print(f"  {m:<12}" + "".join(f"  {r['forward_peak_mb']:>10.2f} MB" for r in row_vals))

    print("\n" + "=" * 92)
    print("PUBMED (TRIPLED FEATURES) — BACKWARD COMPUTATIONAL MEMORY (MB)")
    print("=" * 92)
    print(header)
    print("-" * 92)
    for m in models:
        row_vals = [r for r in all_rows if r["model"] == m]
        print(f"  {m:<12}" + "".join(f"  {r['backward_peak_mb']:>10.2f} MB" for r in row_vals))

    print("\n" + "=" * 92)
    print("PUBMED (TRIPLED FEATURES) — STEP MEMORY (MB)")
    print("=" * 92)
    print(header)
    print("-" * 92)
    for m in models:
        row_vals = [r for r in all_rows if r["model"] == m]
        print(f"  {m:<12}" + "".join(f"  {r['step_peak_mb']:>10.2f} MB" for r in row_vals))

    print("\n" + "=" * 92)
    print("PUBMED (TRIPLED FEATURES) — AVG EPOCH TIME (s)")
    print("=" * 92)
    print(header)
    print("-" * 92)
    for m in models:
        row_vals = [r for r in all_rows if r["model"] == m]
        print(f"  {m:<12}" + "".join(f"  {r['avg_epoch_time_s']:>11.4f}" for r in row_vals))

    print("\n" + "=" * 92)
    print("PUBMED (TRIPLED FEATURES) — BEST TEST ACCURACY")
    print("=" * 92)
    print(header)
    print("-" * 92)
    for m in models:
        row_vals = [r for r in all_rows if r["model"] == m]
        print(f"  {m:<12}" + "".join(f"  {r['best_test_acc']:>11.4f}" for r in row_vals))
    print()


def main():
    args = parse_args()
    dataset, full_data = load_pubmed_tripled(root=args.root, seed=args.seed)

    models = ["graphsage", "graphsaint", "clustergcn"]
    fractions = [0.25, 0.5, 0.75, 1.0]

    print("Loaded PubMed with tripled features")
    print(f"Nodes: {full_data.num_nodes}")
    print(f"Edges: {full_data.num_edges}")
    print(f"Input features: {full_data.num_node_features}")
    print(f"Classes: {dataset.num_classes}")

    all_rows = []

    for model_name in models:
        for fraction in fractions:
            print("\n" + "=" * 78)
            print(f"Running {model_name} | fraction={fraction} | dataset=pubmed_tripled | memory_mode={args.memory_mode}")
            print("=" * 78)

            data = sample_pubmed_subgraph(full_data, fraction=fraction, seed=args.seed)
            model = build_model(
                model_name,
                data.num_node_features,
                args.hidden_dim,
                dataset.num_classes,
                args.dropout,
            )

            summary = _train(model_name, model, data, args.epochs, args.memory_mode)

            if args.memory_mode == "rss":
                fwd = summary.ram_stats.get("forward", {})
                all_rows.append({
                    "model": model_name,
                    "fraction": fraction,
                    "nodes": int(data.num_nodes),
                    "edges": int(data.num_edges),
                    "best_test_acc": summary.best_test_acc,
                    "avg_epoch_time_s": summary.avg_epoch_time,
                    "forward_peak_mb": fwd.get("peak_mb_mean", 0.0),
                    "forward_overhead_mb": fwd.get("overhead_mb_mean", 0.0),
                    "forward_time_ms": fwd.get("duration_s_mean", 0.0) * 1000,
                })
            else:
                cm = summary["computational_memory"]
                all_rows.append({
                    "model": model_name,
                    "fraction": fraction,
                    "nodes": int(data.num_nodes),
                    "edges": int(data.num_edges),
                    "best_test_acc": summary["best_test_acc"],
                    "avg_epoch_time_s": summary["avg_epoch_time"],
                    "forward_peak_mb": cm["forward_peak_mb_mean"],
                    "backward_peak_mb": cm["backward_peak_mb_mean"],
                    "step_peak_mb": cm["step_peak_mb_mean"],
                })

    if args.memory_mode == "rss":
        _print_rss_tables(all_rows)
    else:
        _print_comp_tables(all_rows)


if __name__ == "__main__":
    main()