import sys
import torch
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from typing import Type
from ogb.graphproppred import collate_dgl
from dgl.dataloading import GraphDataLoader

from torch.utils.data import Dataset
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from dtc_gnn.data_management.data_transforms import GraphDataToTensorTransform


class GraphDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            code_dist: int
    ):
        code_dist = "ensemble" if code_dist < 0 else code_dist
        data = np.load(
            file=str(Path(sys.path[1]) / data_dir / f'data_dist_{code_dist}.npz'),
            allow_pickle=True)
        self._x_data = data['graphs']
        self._y_data = data['labels']

        self.graph_data_transform = GraphDataToTensorTransform()
        self.hot_labels = self._labels_to_one_hots()

    def __len__(self):
        return len(self._x_data)

    @staticmethod
    def _labels_to_one_hots():
        labels = ["I", "X", "Z", "Y"]
        one_hots = torch.eye(len(labels), len(labels), dtype=torch.float32)
        return {
            label: encoding for label, encoding in
            zip(sorted(labels), [t for t in one_hots])
        }

    def __getitem__(self, idx):
        x = self._x_data[idx]
        if x is not None:
            x = self.graph_data_transform(x)
        y1, y2 = str(self._y_data[idx])
        y1, y2 = self.hot_labels[y1], self.hot_labels[y2]
        return x, torch.cat([y1, y2], dim=0)


class GraphDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset: Type[GraphDataset],
            data_train_dir: str,
            data_val_dir: str,
            data_test_dir: str,
            code_dist: int,
            batch_size: int
    ):
        super().__init__()
        self._dataset = dataset
        self._data_train_dir = data_train_dir
        self._data_val_dir = data_val_dir
        self._data_test_dir = data_test_dir
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
        return GraphDataLoader(
            dataset=self._dataset(self._data_test_dir, self._code_dist),
            collate_fn=collate_dgl,
            **{'batch_size': self._batch_size,
               'drop_last': False,
               'num_workers': 0}
        )
