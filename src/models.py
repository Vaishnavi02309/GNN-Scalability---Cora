import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, ClusterGCNConv


class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, snapshot_fn=None):
        if snapshot_fn is not None:
            snapshot_fn("forward_start")

        x = self.conv1(x, edge_index)
        if snapshot_fn is not None:
            snapshot_fn("after_conv1")

        x = F.relu(x)
        if snapshot_fn is not None:
            snapshot_fn("after_relu")

        x = F.dropout(x, p=self.dropout, training=self.training)
        if snapshot_fn is not None:
            snapshot_fn("after_dropout")

        x = self.conv2(x, edge_index)
        if snapshot_fn is not None:
            snapshot_fn("after_conv2")

        return x


class ClusterGCNNet(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = ClusterGCNConv(in_channels, hidden_channels)
        self.conv2 = ClusterGCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, snapshot_fn=None):
        if snapshot_fn is not None:
            snapshot_fn("forward_start")

        x = self.conv1(x, edge_index)
        if snapshot_fn is not None:
            snapshot_fn("after_conv1")

        x = F.relu(x)
        if snapshot_fn is not None:
            snapshot_fn("after_relu")

        x = F.dropout(x, p=self.dropout, training=self.training)
        if snapshot_fn is not None:
            snapshot_fn("after_dropout")

        x = self.conv2(x, edge_index)
        if snapshot_fn is not None:
            snapshot_fn("after_conv2")

        return x