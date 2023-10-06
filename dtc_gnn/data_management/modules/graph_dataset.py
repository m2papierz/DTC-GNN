import sys
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


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

    def _labels_onehot_to_single_tensor(self, idx):
        return torch.cat(
            tensors=[
                self.hot_labels[self._y_data[idx][0]],
                self.hot_labels[self._y_data[idx][1]]
            ], dim=0
        )

    def __getitem__(self, idx):
        x = self._x_data[idx]
        y = self._labels_onehot_to_single_tensor(idx)
        return x, y
