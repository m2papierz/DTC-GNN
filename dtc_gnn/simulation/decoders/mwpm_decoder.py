from qecsim.paulitools import bsp
from qecsim.models.toric import ToricMWPMDecoder
from dtc_gnn.simulation.decoders.base import DecoderBase


class DecoderMWPM(DecoderBase):
    def __init__(self):
        super().__init__()

    def _init_decoder(self):
        self._decoder = ToricMWPMDecoder()

    def _predict(self, syndrome, code=None):
        physical_error_pred = self._decoder.decode(code, syndrome)
        logical_error_pred = bsp(physical_error_pred, code.logicals.T)
        return logical_error_pred
