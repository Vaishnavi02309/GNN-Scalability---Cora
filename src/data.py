from __future__ import annotations

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def load_cora(root: str = "data"):
    dataset = Planetoid(root=root, name="Cora")
    return dataset, dataset[0]


def sample_cora_subgraph(data: Data, fraction: float, seed: int = 42) -> Data:
    if fraction >= 0.999:
        return data.clone()
    if fraction <= 0:
        raise ValueError("fraction must be > 0")

    num_nodes = data.num_nodes
    keep = max(1, int(num_nodes * fraction))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)
    node_idx = perm[:keep]
    node_idx, _ = torch.sort(node_idx)

    edge_index, edge_attr = subgraph(node_idx, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)

    new_data = Data(
        x=data.x[node_idx],
        edge_index=edge_index,
        y=data.y[node_idx],
    )

    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(data, mask_name):
            setattr(new_data, mask_name, getattr(data, mask_name)[node_idx])

    # fallback in case a split becomes empty
    if int(new_data.train_mask.sum()) == 0:
        new_data.train_mask = torch.zeros(new_data.num_nodes, dtype=torch.bool)
        new_data.train_mask[: min(20, new_data.num_nodes)] = True
    if int(new_data.val_mask.sum()) == 0:
        new_data.val_mask = torch.zeros(new_data.num_nodes, dtype=torch.bool)
        start = min(20, new_data.num_nodes)
        end = min(start + 20, new_data.num_nodes)
        new_data.val_mask[start:end] = True
    if int(new_data.test_mask.sum()) == 0:
        new_data.test_mask = ~(new_data.train_mask | new_data.val_mask)

    return new_data
