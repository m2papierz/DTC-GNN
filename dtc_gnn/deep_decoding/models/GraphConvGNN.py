import dgl
import torch
import torch.nn as nn
import torch.nn.functional as fn

from dgl.nn import GraphConv


class PredictionLayer(nn.Module):
    def __init__(self, in_dim: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Linear prediction layers
        self.prediction_projection = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU()
        )

        # Prediction for 1st qubit
        self.q1_head = nn.Sequential(
            nn.Linear(in_dim // 2, 2),
            nn.Sigmoid()
        )

        # Prediction for 2nd qubit
        self.q2_head = nn.Sequential(
            nn.Linear(in_dim // 2, 2),
            nn.Sigmoid()
        )

    def forward(self, h):
        h = self.prediction_projection(h)
        q1 = self.q1_head(h)
        q2 = self.q2_head(h)
        return q1, q2


class GraphConvGNN(nn.Module):
    def __init__(
            self,
            n_h_dim: int,
            layers_num: int,
            dropout_rate: float,
    ):
        super(GraphConvGNN, self).__init__()

        # Projection layers
        self.n_projection = nn.Linear(in_features=3, out_features=n_h_dim // 4)

        # Convolutional layers
        self.conv_layers = nn.ModuleList(
            [GraphConv(in_feats=n_h_dim // 4, out_feats=n_h_dim)])
        for _ in range(layers_num):
            self.conv_layers.append(
                GraphConv(in_feats=n_h_dim, out_feats=n_h_dim))
        self.conv_layers.append(
            GraphConv(in_feats=n_h_dim, out_feats=n_h_dim // 4))

        # Dropout regularization
        self.h_dropout = nn.Dropout(dropout_rate)

        # Linear prediction layers
        self.prediction = PredictionLayer(
            in_dim=n_h_dim // 4, dropout_rate=dropout_rate)

    def forward(self, g=None, h=None, e=None, error=True):
        if not error:
            identity = torch.Tensor([[0, 0]])
            return identity, identity

        # Nodes features projection
        h = self.n_projection(h)
        h = self.h_dropout(h)
        h = fn.leaky_relu(h)

        # Graph convolution layers
        for conv_layer in self.conv_layers:
            h = conv_layer(g, h, edge_weight=e)
            h = self.h_dropout(h)
            h = fn.leaky_relu(h)

        # Node features mean-pooling
        g.ndata['hm'] = h
        h = dgl.mean_nodes(g, feat='hm')

        return self.prediction(h)
