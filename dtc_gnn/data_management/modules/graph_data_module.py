import pytorch_lightning as pl

from typing import Type
from ogb.graphproppred import collate_dgl
from dgl.dataloading import GraphDataLoader

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from dtc_gnn.data_management.modules.graph_dataset import GraphDataset


class GraphDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset: Type[GraphDataset],
            data_train_dir: str,
            data_val_dir: str,
            code_dist: int,
            batch_size: int
    ):
        super().__init__()
        self._dataset = dataset
        self._data_train_dir = data_train_dir
        self._data_val_dir = data_val_dir
        self._code_dist = code_dist
        self._batch_size = batch_size

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return GraphDataLoader(
            dataset=self._dataset(self._data_train_dir, self._code_dist),
            collate_fn=collate_dgl,
            **{'batch_size': self._batch_size,
               'drop_last': False,
               'num_workers': 0}
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return GraphDataLoader(
            dataset=self._dataset(self._data_val_dir, self._code_dist),
            collate_fn=collate_dgl,
            **{'batch_size': self._batch_size,
               'drop_last': False,
               'num_workers': 0}
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return NotImplementedError
