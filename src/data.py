from __future__ import annotations

from collections import deque

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph


def load_cora(root: str = "data"):
    """
    Load the Cora citation network.

    Returns:
        dataset: The PyG Planetoid dataset wrapper.
        data: The single graph object from the dataset.
    """
    dataset = Planetoid(root=root, name="Cora")
    return dataset, dataset[0]


def _build_adjacency_list(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """
    Build an undirected adjacency list from edge_index.

    We treat the graph as undirected for expansion so that seed growth
    can move naturally through the neighborhood structure.
    """
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for u, v in zip(src, dst):
        if u == v:
            neighbors[u].add(v)
        else:
            neighbors[u].add(v)
            neighbors[v].add(u)

    return [sorted(list(nbrs)) for nbrs in neighbors]


def build_seed_growth_order(
    data: Data,
    num_seed_nodes: int = 6,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a deterministic node growth order starting from a small set of seed nodes
    and expanding outward with BFS-style neighborhood growth.

    This gives one global order of nodes:
        [seed nodes, their neighbors, next-hop neighbors, ...]

    Fractions such as 25%, 50%, 75% are then created by taking prefixes of this order,
    which guarantees nested graphs:
        25% ⊂ 50% ⊂ 75% ⊂ 100%

    Args:
        data: Full PyG graph.
        num_seed_nodes: Number of starting seed nodes.
        seed: Random seed used only for choosing the initial seeds.

    Returns:
        growth_order: Tensor of original node ids in the order they were added.
        seed_nodes: Tensor of the initial chosen seed node ids.
    """
    num_nodes = int(data.num_nodes)
    if num_nodes <= 0:
        raise ValueError("Graph must contain at least one node.")
    if num_seed_nodes <= 0:
        raise ValueError("num_seed_nodes must be > 0.")

    num_seed_nodes = min(num_seed_nodes, num_nodes)

    g = torch.Generator().manual_seed(seed)
    seed_nodes = torch.randperm(num_nodes, generator=g)[:num_seed_nodes]
    seed_nodes, _ = torch.sort(seed_nodes)

    adjacency = _build_adjacency_list(data.edge_index, num_nodes)

    visited = torch.zeros(num_nodes, dtype=torch.bool)
    growth_order: list[int] = []
    queue: deque[int] = deque()

    # Start from the chosen seed nodes.
    for node in seed_nodes.tolist():
        if not visited[node]:
            visited[node] = True
            growth_order.append(node)
            queue.append(node)

    # BFS-style outward expansion.
    while len(growth_order) < num_nodes:
        if queue:
            current = queue.popleft()
            for nbr in adjacency[current]:
                if not visited[nbr]:
                    visited[nbr] = True
                    growth_order.append(nbr)
                    queue.append(nbr)
        else:
            # Fallback for disconnected components:
            # start a new BFS from the smallest unvisited node.
            remaining = torch.nonzero(~visited, as_tuple=False).view(-1)
            if remaining.numel() == 0:
                break
            next_start = int(remaining[0].item())
            visited[next_start] = True
            growth_order.append(next_start)
            queue.append(next_start)

    growth_order_tensor = torch.tensor(growth_order, dtype=torch.long)

    if growth_order_tensor.numel() != num_nodes:
        raise RuntimeError(
            f"Growth order should contain exactly {num_nodes} nodes, "
            f"but got {growth_order_tensor.numel()}."
        )

    return growth_order_tensor, seed_nodes


def subgraph_from_growth_order(
    data: Data,
    growth_order: torch.Tensor,
    fraction: float,
) -> Data:
    """
    Build a nested induced subgraph by taking the first `fraction` of nodes
    from a precomputed growth order.

    Args:
        data: Full PyG graph.
        growth_order: Original node ids in progressive seed-expansion order.
        fraction: Fraction of nodes to keep.

    Returns:
        new_data: Induced subgraph with relabeled nodes and subset masks.
    """
    if fraction <= 0:
        raise ValueError("fraction must be > 0")
    if fraction >= 0.999:
        return data.clone()

    num_nodes = int(data.num_nodes)
    keep = max(1, int(num_nodes * fraction))

    node_idx = growth_order[:keep]
    node_idx, _ = torch.sort(node_idx)

    edge_index, edge_attr = subgraph(
        node_idx,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )

    new_data = Data(
        x=data.x[node_idx],
        edge_index=edge_index,
        y=data.y[node_idx],
    )

    # Preserve any available node-level masks.
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(data, mask_name):
            setattr(new_data, mask_name, getattr(data, mask_name)[node_idx])

    # Safety fallback in case a split becomes empty after subgraphing.
    # This should be rare, but keeping it prevents training/evaluation failures.
    if hasattr(new_data, "train_mask") and int(new_data.train_mask.sum()) == 0:
        new_data.train_mask = torch.zeros(new_data.num_nodes, dtype=torch.bool)
        new_data.train_mask[: min(20, new_data.num_nodes)] = True

    if hasattr(new_data, "val_mask") and int(new_data.val_mask.sum()) == 0:
        new_data.val_mask = torch.zeros(new_data.num_nodes, dtype=torch.bool)
        start = min(20, new_data.num_nodes)
        end = min(start + 20, new_data.num_nodes)
        new_data.val_mask[start:end] = True

    if hasattr(new_data, "test_mask") and int(new_data.test_mask.sum()) == 0:
        new_data.test_mask = ~(new_data.train_mask | new_data.val_mask)

    return new_data


def sample_cora_subgraph(
    data: Data,
    fraction: float,
    seed: int = 42,
    num_seed_nodes: int = 6,
) -> Data:
    """
    Backward-compatible wrapper.

    This now performs seeded progressive graph growth instead of independent
    random node sampling.

    Args:
        data: Full PyG graph.
        fraction: Target fraction of nodes.
        seed: Random seed for initial seed-node selection.
        num_seed_nodes: Number of initial starting nodes.

    Returns:
        Induced nested subgraph grown from the same seed set.
    """
    growth_order, _ = build_seed_growth_order(
        data=data,
        num_seed_nodes=num_seed_nodes,
        seed=seed,
    )
    return subgraph_from_growth_order(
        data=data,
        growth_order=growth_order,
        fraction=fraction,
    )