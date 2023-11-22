import numpy as np
from typing import Tuple


class GraphNode:
    def __init__(
            self,
            indices: tuple,
            code_dist: int,
            is_x: bool,
            n_counts_x: float,
            n_counts_z: float
    ):
        self._indices = indices
        self._code_dist = code_dist
        self._is_x_stab = is_x
        self._n_counts_x = n_counts_x
        self._n_counts_z = n_counts_z

    @property
    def position(self) -> Tuple[int, int]:
        return self._indices[0], self._indices[1]

    @property
    def features(self) -> Tuple[int, float, float, float, float, float]:
        return (
            1 if self._is_x_stab else 0,
            round(self.position[1] / self._code_dist, 5),
            round(self.position[0] / self._code_dist, 5),
            round(self._n_counts_x, 5),
            round(self._n_counts_z, 5),
            round(self._n_counts_x + self._n_counts_z, 5)
        )


class GraphEdge:
    def __init__(
            self,
            ids: tuple,
            node_a: tuple,
            node_b: tuple
    ):
        self._a_idx, self._b_idx = ids
        self._a_pos = node_a
        self._b_pos = node_b

    @property
    def weight(self) -> float:
        return round(np.power(1 / np.max(
            [np.abs(self._a_pos[0] - self._b_pos[0]) +
             np.abs(self._a_pos[1] - self._b_pos[1])]
        ), 2), 5)

    @property
    def a_idx(self) -> int:
        return self._a_idx

    @property
    def b_idx(self) -> int:
        return self._b_idx
