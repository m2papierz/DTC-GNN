import numpy as np

from dgl import DGLGraph
from typing import List, Union
from qecsim.paulitools import bsp

from dtc_gnn.error_models import ErrorModel
from dtc_gnn.simulation.decoders.base import DecoderBase

from qecsim.models.rotatedtoric import RotatedToricCode
from qecsim.models.rotatedtoric import RotatedToricSMWPMDecoder


class DecoderMWPM(DecoderBase):
    def __init__(self):
        super().__init__()

    def _init_decoder(self):
        self._decoder = RotatedToricSMWPMDecoder()

    @property
    def name(self):
        return "MWPM"

    def _predict(
            self,
            syndrome: Union[List[int], DGLGraph],
            code: RotatedToricCode = None,
            error_model: ErrorModel = None
    ) -> np.ndarray:
        physical_error_pred = self._decoder.decode(
            code, syndrome, error_model)
        logical_error_pred = bsp(
            physical_error_pred, code.logicals.T)
        return logical_error_pred
