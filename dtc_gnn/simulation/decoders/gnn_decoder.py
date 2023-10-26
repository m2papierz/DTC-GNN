import torch
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from dgl import DGLGraph
from typing import List, Union
from dtc_gnn.simulation.decoders.base import DecoderBase


class DecoderGNN(DecoderBase):
    def __init__(self, model_path: Path, torch_module: pl.LightningModule):
        self._model_path = model_path
        self._decoder = torch_module
        super().__init__()

    def _init_decoder(self):
        model_dict = torch.load(self._model_path)
        state_dict = model_dict['state_dict']
        self._decoder.load_state_dict(state_dict)

    @staticmethod
    def _prob_to_pred(
            input_tensor: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(data=[
            1 if x > 0.9 else 0 for x in input_tensor[0]
        ], dtype=torch.float)

    def _predict(
            self,
            syndrome: Union[List[int], DGLGraph],
            code=None
    ) -> np.ndarray:
        if syndrome is not None:
            pred = self._decoder(
                syndrome, syndrome.ndata["feat"], syndrome.edata["weight"])
            pred = self._prob_to_pred(pred)
        else:
            pred = self._decoder(None, None, None, False)
        return pred.detach().numpy()
