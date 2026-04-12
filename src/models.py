from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import SAGEConv

from .profiling import ForwardMemoryTracer


class GraphSAGENet(Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, tracer: Optional[ForwardMemoryTracer] = None) -> Tensor:
        if tracer:
            tracer.record("forward_start")
        x = self.conv1(x, edge_index)
        if tracer:
            tracer.record("after_conv1")
        x = F.relu(x)
        if tracer:
            tracer.record("after_relu")
        x = F.dropout(x, p=self.dropout, training=self.training)
        if tracer:
            tracer.record("after_dropout")
        x = self.conv2(x, edge_index)
        if tracer:
            tracer.record("after_conv2")
        return x


def build_model(name: str, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
    name = name.lower()
    if name in {"graphsage", "graphsaint", "clustergcn"}:
        return GraphSAGENet(in_channels, hidden_channels, out_channels, dropout)
    raise ValueError(f"Unknown model: {name}")
