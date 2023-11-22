import dgl
import torch
import torch.nn as nn
import torch.nn.functional as fn

from dgl.nn import EGATConv


class PredictionLayer(nn.Module):
    def __init__(self, in_dim: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h_dropout = nn.Dropout(dropout_rate)

        # Linear prediction layers
        self.first_projection = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU()
        )

        self.second_projection = nn.Sequential(
            nn.Linear(in_dim // 4, in_dim // 8),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU()
        )

        # Prediction for 1st qubit
        self.prediction_1 = nn.Sequential(
            nn.Linear(in_dim // 8, 2),
            nn.Sigmoid()
        )

        # Prediction for 2nd qubit
        self.prediction_2 = nn.Sequential(
            nn.Linear(in_dim // 8, 2),
            nn.Sigmoid()
        )

    def forward(self, h):
        h = self.first_projection(h)
        h = self.h_dropout(h)
        h = self.second_projection(h)
        h = self.h_dropout(h)
        return self.prediction_1(h), self.prediction_2(h)


class EGATConvGNN(nn.Module):
    def __init__(
            self,
            n_h_dim: int,
            e_h_dim: int,
            n_heads: int,
            layers_num: int,
            dropout_rate: float
    ):
        super(EGATConvGNN, self).__init__()

        # Projection layers
        self.n_projection_h = nn.Linear(in_features=6, out_features=n_h_dim // 2)
        self.n_projection_e = nn.Linear(in_features=1, out_features=e_h_dim // 2)

        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            EGATConv(n_h_dim // 2, e_h_dim // 2, n_h_dim, e_h_dim, n_heads)])
        for _ in range(layers_num):
            self.conv_layers.append(
                EGATConv(
                    n_h_dim * n_heads, e_h_dim * n_heads, n_h_dim, e_h_dim, n_heads)
            )
        self.conv_layers.append(
            EGATConv(
                n_h_dim * n_heads, e_h_dim * n_heads, n_h_dim * n_heads,  e_h_dim * n_heads, 1)
        )

        # Dropout regularization
        self.h_dropout = nn.Dropout(dropout_rate)

        # Linear prediction layers
        self.prediction = PredictionLayer(
            in_dim=n_h_dim * n_heads, dropout_rate=dropout_rate)

    def forward(self, g=None, h=None, e=None, error=True):
        if not error:
            identity = torch.Tensor([[0, 0]])
            return identity, identity

        # Nodes features projection
        h = self.n_projection_h(h)
        e = self.n_projection_e(e)
        h = self.h_dropout(h)
        h = fn.leaky_relu(h)

        # Graph convolution layers
        bsh, bse = h.shape[0], e.shape[0]
        for conv_layer in self.conv_layers:
            h, e = conv_layer(g, h, e)
            h = h.reshape(bsh, -1)
            e = e.reshape(bse, -1)
            h = self.h_dropout(h)
            e = self.h_dropout(e)
            h = fn.leaky_relu(h)

        # Node features mean-pooling
        g.ndata['hm'] = h
        h = dgl.mean_nodes(g, feat='hm')

        return self.prediction(h)
