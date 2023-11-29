import sys

import numpy as np
import qecsim.models.rotatedtoric
import qecsim.paulitools as pt

from tqdm import tqdm
from typing import Union, List, Type
from dtc_gnn.simulation.decoders.mwpm_decoder import DecoderMWPM
from dtc_gnn.simulation.decoders.gnn_decoder import DecoderGNN
from dtc_gnn.data_management.transforms.graph_syndrome import GraphSyndromeTransform
from dtc_gnn.error_models import ErrorModel


class Simulator:
    def __init__(
            self,
            stabilizer_code: Type[qecsim.models.rotatedtoric.RotatedToricCode],
            error_model: ErrorModel,
            graph_transform: GraphSyndromeTransform,
            num_shots: int,
            code_distances: List[int],
            max_error_prob: float,
            prob_space_num: int
    ):
        self._stab_code = stabilizer_code
        self._error_model = error_model
        self._graph_transform = graph_transform

        self._num_shots = num_shots
        self._code_dist = code_distances
        self._prob_linspace = np.linspace(
            start=0.0, stop=max_error_prob, num=prob_space_num)

    def _errors_generator(self, code, p):
        for _ in range(self._num_shots):
            error = self._error_model.generate(
                shape=code.n_k_d[0], probability=p)
            syndrome = pt.bsp(error, code.stabilizers.T)
            log_error = pt.bsp(error, code.logicals.T)

            if np.any(syndrome):
                graph = self._graph_transform(code, syndrome)
            else:
                graph = None

            yield graph, syndrome, log_error

    def _failures_via_physical_frame_changes(self, decoder, code_dist, error_prob):
        num_errors = 0
        stab_code = self._stab_code(rows=code_dist, columns=code_dist)
        generator = self._errors_generator(stab_code, error_prob)

        desc = f'Gathering errors data for L={code_dist}, p={round(error_prob, 3)}'
        for g, s, l_e in tqdm(generator, total=self._num_shots, desc=desc, file=sys.stdout):

            if isinstance(decoder, DecoderMWPM):
                log_pred = decoder.decode(
                    syndrome=s,
                    code=stab_code,
                    error_model=self._error_model)
            else:
                log_pred = decoder.decode(syndrome=g)

            if not np.array_equal(log_pred, l_e):
                num_errors += 1

        return num_errors

    def run_simulation(self, decoder: Union[DecoderMWPM, DecoderGNN]):
        log_errors_code_dist = []
        for L in self._code_dist:
            log_errors = []
            for p in self._prob_linspace:
                num_errors = self._failures_via_physical_frame_changes(
                    decoder=decoder, code_dist=L, error_prob=p)
                log_errors.append(num_errors / self._num_shots)
            log_errors_code_dist.append(np.array(log_errors))

        return log_errors_code_dist
