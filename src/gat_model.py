from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SimpleGATNet(nn.Module):
    """
    Two-layer GAT baseline using PyG GATConv.

    This is an edge-aware local-attention GNN:
    - attention is computed only over graph edges
    - not global all-pairs attention
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        self.dropout = dropout
        self.heads = heads

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_activations: bool = False,
    ):
        x_in = x

        x1, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        x1_relu = F.elu(x1)
        x1_drop = F.dropout(x1_relu, p=self.dropout, training=self.training)

        x2, attn2 = self.gat2(x1_drop, edge_index, return_attention_weights=True)

        if return_activations:
            edge_index1, alpha1 = attn1
            edge_index2, alpha2 = attn2

            return x2, {
                "input_x": x_in,
                "layer1_out": x1,
                "layer1_relu": x1_relu,
                "layer1_drop": x1_drop,
                "layer2_out": x2,
                "attn_edge_index1": edge_index1,
                "attn_alpha1": alpha1,
                "attn_edge_index2": edge_index2,
                "attn_alpha2": alpha2,
                "logits": x2,
            }

        return x2