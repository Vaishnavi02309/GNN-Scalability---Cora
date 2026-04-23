from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pubmed_tripled(root: str = "data", seed: int = 42):
    """
    Load PubMed and expand features by concatenating them 3 times.

    Original PubMed:
        num_nodes    ≈ 19,717
        num_edges    ≈ 88,648
        num_features = 500

    Tripled version:
        num_features = 1500
    """
    set_seed(seed)

    dataset = Planetoid(root=root, name="PubMed")
    data = dataset[0]

    # Triplicate features: [N, 500] -> [N, 1500]
    data.x = torch.cat([data.x, data.x, data.x], dim=1)

    return dataset, data


def sample_pubmed_subgraph(full_data, fraction: float = 1.0, seed: int = 42):
    """
    Induced random-node subgraph, same style as your Cora fraction experiments.
    Keeps masks/labels/features aligned to the sampled nodes.
    """
    set_seed(seed)

    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    num_nodes = full_data.num_nodes
    keep_n = max(2, int(round(num_nodes * fraction)))

    perm = torch.randperm(num_nodes)
    node_idx = perm[:keep_n]
    node_idx, _ = torch.sort(node_idx)

    edge_index, edge_attr = subgraph(
        subset=node_idx,
        edge_index=full_data.edge_index,
        edge_attr=None,
        relabel_nodes=True,
        num_nodes=num_nodes,
        return_edge_mask=False,
    )

    data = full_data.__class__()
    data.x = full_data.x[node_idx]
    data.y = full_data.y[node_idx]
    data.edge_index = edge_index

    if hasattr(full_data, "train_mask"):
        data.train_mask = full_data.train_mask[node_idx]
    if hasattr(full_data, "val_mask"):
        data.val_mask = full_data.val_mask[node_idx]
    if hasattr(full_data, "test_mask"):
        data.test_mask = full_data.test_mask[node_idx]

    return data