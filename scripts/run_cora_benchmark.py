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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    dataset, full_data = load_cora(root=os.path.join(ROOT, "data"))
    models = ["graphsage", "graphsaint", "clustergcn"]
    fractions = [0.25, 0.5, 0.75, 1.0]

    all_rows = []
    for model_name in models:
        for fraction in fractions:
            print("\n" + "=" * 72)
            print(f"Running {model_name} with fraction={fraction}")
            print("=" * 72)
            data = sample_cora_subgraph(full_data, fraction=fraction, seed=args.seed)
            model = build_model(model_name, data.num_node_features, args.hidden_dim, dataset.num_classes, args.dropout)
            res = train_full_batch(model, data, epochs=args.epochs)
            row = {
                "model": model_name,
                "fraction": fraction,
                "nodes": int(data.num_nodes),
                "edges": int(data.num_edges),
                "best_val_acc": res["best"]["val_acc"],
                "best_test_acc": res["best"]["test_acc"],
                "best_test_macro_f1": res["best"]["test_macro_f1"],
                "avg_epoch_time_s": res["avg_epoch_time_s"],
                "max_rss_after_forward_mb": res["max_rss_after_forward_mb"],
                "max_rss_after_backward_mb": res["max_rss_after_backward_mb"],
            }
            all_rows.append(row)
            print(row)

    print("\nSummary")
    for row in all_rows:
        print(row)


if __name__ == "__main__":
    main()
