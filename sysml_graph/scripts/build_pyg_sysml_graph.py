import csv
import torch
from pathlib import Path
from torch_geometric.data import Data

OUTPUT_DIR = Path("sysml_graph/output")


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    nodes = read_csv(OUTPUT_DIR / "nodes.csv")
    edges = read_csv(OUTPUT_DIR / "edges.csv")

    node_to_idx = {node["node_id"]: idx for idx, node in enumerate(nodes)}

    edge_pairs = []
    edge_types_raw = []

    for edge in edges:
        src = edge["source_id"]
        tgt = edge["target_id"]

        if src in node_to_idx and tgt in node_to_idx:
            edge_pairs.append([node_to_idx[src], node_to_idx[tgt]])
            edge_types_raw.append(edge["edge_type"])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    node_types = sorted(set(node["node_type"] for node in nodes))
    node_type_to_idx = {node_type: idx for idx, node_type in enumerate(node_types)}

    # Simple node features: one-hot encoding of SysML node type
    x = torch.zeros((len(nodes), len(node_types)), dtype=torch.float)

    for idx, node in enumerate(nodes):
        x[idx, node_type_to_idx[node["node_type"]]] = 1.0

    edge_types = sorted(set(edge_types_raw))
    edge_type_to_idx = {edge_type: idx for idx, edge_type in enumerate(edge_types)}

    if edge_types_raw:
        edge_type_tensor = torch.tensor(
            [edge_type_to_idx[t] for t in edge_types_raw],
            dtype=torch.long
        )
    else:
        edge_type_tensor = torch.empty((0,), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type_tensor
    )

    torch.save(
        {
            "data": data,
            "node_to_idx": node_to_idx,
            "node_types": node_types,
            "edge_types": edge_types,
        },
        OUTPUT_DIR / "sysml_graph.pt"
    )

    print(data)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Node feature dimension: {x.shape[1]}")
    print(f"Number of node types: {len(node_types)}")
    print(f"Number of edge types: {len(edge_types)}")


if __name__ == "__main__":
    main()
