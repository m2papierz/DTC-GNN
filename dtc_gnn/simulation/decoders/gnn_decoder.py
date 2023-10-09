import torch
import pytorch_lightning as pl

from pathlib import Path
from qecsim.paulitools import pauli_to_bsf
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
    def _prob_to_pred(input_tensor):
        return torch.tensor([float(i == input_tensor.max()) for i in input_tensor[0]])

    def _predict(self, syndrome, code=None):
        if syndrome is not None:
            pred_q1, pred_q2 = self._decoder(
                syndrome, syndrome.ndata["feat"], syndrome.edata["weight"])
        else:
            pred_q1, pred_q2 = self._decoder(None, None, None, False)
        pred_q1 = self._prob_to_pred(pred_q1)
        pred_q2 = self._prob_to_pred(pred_q2)
        return torch.cat(tensors=[pred_q1, pred_q2], dim=0)
