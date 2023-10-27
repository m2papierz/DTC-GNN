import abc
import numpy as np

from typing import Union, Tuple, List
from qecsim.paulitools import pauli_to_bsf


class ErrorModel:
    paulis = ('I', 'X', 'Y', 'Z')
    rng = np.random.default_rng()

    @abc.abstractmethod
    def probability_distribution(self, probability: float):
        pass

    def generate(
            self,
            shape: Union[int, Tuple[int, int]],
            probability: float
    ) -> Union[np.ndarray, List[np.ndarray]]:
        assert isinstance(shape, (int, tuple)), \
            "Shape can be integer or tuple only"

        if isinstance(shape, tuple):
            pauli_errors = self.rng.choice(
                self.paulis, size=shape,
                p=self.probability_distribution(probability)
            )
            pauli_errors = [''.join(e) for e in pauli_errors]
        else:
            pauli_errors = ''.join(self.rng.choice(
                self.paulis, size=shape,
                p=self.probability_distribution(probability)
            ))
        return pauli_to_bsf(pauli_errors)

    def __repr__(self):
        return '{}()'.format(type(self).__name__)


class DepolarizingErrorModel(ErrorModel):
    """
    Implements a depolarizing error model.

    The probability distribution for a given error probability p is:
    (1 - p) for no error and p/3 for X, Y, Z errors.
    """
    def probability_distribution(self, probability):
        p_x = p_y = p_z = probability / 3
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z


class BitFlipErrorModel(ErrorModel):
    """
    Implements a bit-flip error model.

    The probability distribution for a given error probability p is:
    (1 - p) for no error and p for X error.
    """
    def probability_distribution(self, probability):
        p_x = probability
        p_y = p_z = 0
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z


class PhaseFlipErrorModel(ErrorModel):
    """
    Implements a phase-flip error model.

    The probability distribution for a given error probability p is:
    (1 - p) for no error and p for Z error.
    """
    def probability_distribution(self, probability):
        p_x = p_y = 0
        p_z = probability
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z


class BitPhaseFlipErrorModel(ErrorModel):
    """
    Implements a bit-phase-flip error model.

    The probability distribution for a given error probability p is:
    (1 - p) for no error and p for Y error.
    """
    def probability_distribution(self, probability):
        p_x = 0
        p_y = probability
        p_z = 0
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z
