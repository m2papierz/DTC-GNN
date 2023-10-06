import numpy as np
from typing import Tuple


class GraphNode:
    def __init__(self, indices: tuple, code_dist: int):
        self._indices = indices
        self._code_dist = code_dist
        self._is_x_stab = indices[0] == 1

    @property
    def position(self) -> Tuple[int, int]:
        if self._is_x_stab:
            return self._indices[2], self._indices[1]
        else:
            return self._indices[2] + 0.5, self._indices[1] - 0.5

    @property
    def features(self) -> Tuple[int, float, float]:
        return (
            1 if self._is_x_stab else 0,
            self.position[1] / self._code_dist,
            self.position[0] / self._code_dist
        )


class GraphEdge:
    def __init__(self, ids: tuple, node_a: tuple, node_b: tuple):
        self._a_idx, self._b_idx = ids
        self._a_pos = node_a
        self._b_pos = node_b

    @property
    def weight(self) -> float:
        return np.power(1 / np.max(
            [np.abs(self._a_pos[0] - self._b_pos[0]) +
             np.abs(self._a_pos[1] - self._b_pos[1])]
        ), 2)

    @property
    def a_idx(self) -> int:
        return self._a_idx

    @property
    def b_idx(self) -> int:
        return self._b_idx
