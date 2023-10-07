import sys
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from dtc_gnn.data_management.label_encoder import LabelEncoder


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

        self._label_encoder = LabelEncoder()

    def __len__(self):
        return len(self._x_data)

    def _labels_onehot_to_single_tensor(self, idx):
        return torch.cat(
            tensors=[
                self._label_encoder.encode(self._y_data[idx][0]),
                self._label_encoder.encode(self._y_data[idx][1])
            ], dim=0
        )

    def __getitem__(self, idx):
        x = self._x_data[idx]
        y = self._labels_onehot_to_single_tensor(idx)
        return x, y
