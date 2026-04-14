from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ClusterGCNConv, SAGEConv


class GraphSAGENet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class ClusterGCNNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = ClusterGCNConv(in_channels, hidden_channels)
        self.conv2 = ClusterGCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def build_model(name: str, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
    key = name.lower()
    if key in {"graphsage", "graphsaint"}:
        return GraphSAGENet(in_channels, hidden_channels, out_channels, dropout=dropout)
    if key == "clustergcn":
        return ClusterGCNNet(in_channels, hidden_channels, out_channels, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")