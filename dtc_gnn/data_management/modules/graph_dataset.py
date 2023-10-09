import sys
import torch
import numpy as np

from pathlib import Path
from qecsim.paulitools import pauli_to_bsf
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

    def __len__(self):
        return len(self._x_data)

    def __getitem__(self, idx):
        x = self._x_data[idx]
        y = torch.tensor(
            pauli_to_bsf(self._y_data[idx]), dtype=torch.float32)
        return x, y
