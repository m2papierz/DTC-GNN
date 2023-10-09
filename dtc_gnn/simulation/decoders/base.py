from dgl import DGLGraph
from typing import List, Union
from abc import abstractmethod


class DecoderBase:
    def __init__(self):
        self._init_decoder()

    @abstractmethod
    def _init_decoder(self):
        pass

    @abstractmethod
    def _predict(self, syndrome: Union[List[int], DGLGraph], code=None):
        pass

    def decode(self, syndrome: Union[List[int], DGLGraph], code=None):
        return self._predict(syndrome, code)
