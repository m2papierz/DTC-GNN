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

    def forward(self, graph, nodes, edges):
        return self._gnn_model(graph, nodes, edges)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, mode='min', patience=self._reduce_lr, min_lr=1e-6
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def _unpack_labels(batch_labels):
        labels_qubit_1 = batch_labels[:, [0, 1, 2, 3]]
        labels_qubit_2 = batch_labels[:, [4, 5, 6, 7]]
        return labels_qubit_1, labels_qubit_2

    def _shared_step(self, batch):
        x = batch[0]
        y1, y2 = self._unpack_labels(batch[1])
        pred_q1, pred_q2 = self(x, x.ndata["feat"], x.edata["weight"])

        loss_q1 = self._loss_function(pred_q1, y1)
        loss_q2 = self._loss_function(pred_q2, y2)
        loss = (loss_q1 + loss_q2) / 2

        return loss, pred_q1, pred_q2

    def training_step(self, batch, batch_idx):
        train_loss, _, _ = self._shared_step(batch)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        y1, y2 = self._unpack_labels(batch[1])
        val_loss, pred1, pred2 = self._shared_step(batch)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for eval_data in [("q1", pred1, y1), ("q2", pred2, y2)]:
            for metric_elem in self._val_metrics:
                metric_elem.to(self.device)
                self.log(
                    f"val_{eval_data[0]}_" + metric_elem.__class__.__name__,
                    metric_elem(eval_data[1], eval_data[2]),
                    on_epoch=True,
                    prog_bar=False,
                )

    def test_step(self, batch, batch_idx):
        y1, y2 = self._unpack_labels(batch[1])
        test_loss, pred1, pred2 = self._shared_step(batch)

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        for eval_data in [("q1", pred1, y1), ("q2", pred2, y2)]:
            for metric_elem in self._val_metrics:
                metric_elem.to(self.device)
                self.log(
                    f"val_{eval_data[0]}_" + metric_elem.__class__.__name__,
                    metric_elem(eval_data[1], eval_data[2]),
                    on_epoch=True,
                    prog_bar=False,
                )
