from dgl import DGLGraph
from typing import List, Union
from abc import abstractmethod

from dtc_gnn.error_models import ErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode


class DecoderBase:
    def __init__(self):
        self._init_decoder()

    @abstractmethod
    def _init_decoder(self):
        pass

    @abstractmethod
    def _predict(
            self,
            syndrome: Union[List[int], DGLGraph],
            code: RotatedPlanarCode = None,
            error_model: ErrorModel = None
    ):
        pass

    def decode(
            self,
            syndrome: Union[List[int], DGLGraph],
            code: RotatedPlanarCode = None,
            error_model: ErrorModel = None
    ):
        return self._predict(syndrome, code, error_model)
