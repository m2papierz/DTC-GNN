import torch
import torchmetrics
import pytorch_lightning as pl

from typing import List
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GNNLightningModule(pl.LightningModule):
    def __init__(
            self,
            torch_module: torch.nn.Module,
            loss_function: torch.nn.Module,
            learning_rate: float,
            reduce_lr_patience: int,
            train_metrics: List[torchmetrics.Metric],
            val_metrics: List[torchmetrics.Metric],
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._gnn_model = torch_module

        self._loss_function = loss_function
        self._learning_rate = learning_rate
        self._reduce_lr = reduce_lr_patience

        self._train_metrics = train_metrics
        self._val_metrics = val_metrics

    def forward(self, graph, nodes, edges, error=True):
        return self._gnn_model(graph, nodes, edges, error)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, mode='min', patience=self._reduce_lr,
                factor=0.5, min_lr=1e-6, verbose=True
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def _shared_step(self, batch):
        x, y = batch[0], batch[1]
        pred = self(x, x.ndata["feat"], x.edata["weight"])
        return self._loss_function(pred, y)

    def training_step(self, batch, batch_idx):
        train_loss = self._shared_step(batch)
        self.log(
            "train_loss", train_loss,
            on_step=False, on_epoch=True, prog_bar=True
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._shared_step(batch)
        self.log(
            "val_loss", val_loss,
            on_step=False, on_epoch=True, prog_bar=True
        )
