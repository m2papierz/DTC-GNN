import sys
import glob
import random
import numpy as np
import qecsim.models.rotatedtoric
import qecsim.paulitools as pt

from tqdm import trange
from pathlib import Path
from typing import List, Tuple, Dict, Type


def random_split(
        input_list: List[any],
        split_ratio: float,
) -> Tuple[List[any], List[any]]:
    """
    Returns input list split into two subsets with given split ratio.
    """
    assert 0 < split_ratio < 1.0, "split_ratio must be in (0, 1.0)"

    random.shuffle(input_list)
    split_index = int(np.floor(split_ratio * len(input_list)))

    subset_a = input_list[:split_index]
    subset_b = input_list[split_index:]

    return subset_a, subset_b


class GraphDataGenerator:
    def __init__(
            self,
            data_directories: Dict[str, str],
            stabilizer_code: Type[qecsim.models.rotatedtoric.RotatedToricCode],
            error_model: any,
            graph_transform: any,
            n_samples: int,
            split_ratio: float,
            code_distances: List[int],
            error_probabilities: List[float]
    ):

        self._stab_code_ = stabilizer_code
        self._error_model = error_model
        self._graph_transform = graph_transform
        self._n_samples = n_samples
        self._split_ratio = split_ratio
        self._code_distances = code_distances
        self._error_probas = error_probabilities

        self.data_dirs = {
            k: Path(sys.path[1]) / v for k, v in data_directories.items()
        }

    def _prob_data_generator(self, size, code, p):
        desc = f"Generating data for code_dist={code.n_k_d[-1]} and p={p}"
        for idx in trange(size, file=sys.stdout, desc=desc):
            error, syndrome = [], []

            # with no error/syndrome graph cannot be created
            while not np.any(error) or not np.any(syndrome):
                error = self._error_model.generate(
                    shape=code.n_k_d[0], probability=p)
                syndrome = pt.bsp(error, code.stabilizers.T)

            graph = self._graph_transform(code, syndrome)
            label = pt.bsf_to_pauli(pt.bsp(error, code.logicals.T))

            yield idx, graph, label

    def _generate_single_code_data(self, code_dist):
        p_samples = int(self._n_samples / len(self._error_probas))
        code = self._stab_code_(rows=code_dist, columns=code_dist)

        graphs_code = np.empty(shape=self._n_samples, dtype=object)
        labels_code = np.empty(shape=self._n_samples, dtype=object)
        for idx, p in enumerate(self._error_probas):
            graphs_prob = np.empty(shape=p_samples, dtype=object)
            labels_prob = np.empty(shape=p_samples, dtype=object)
            for jdx, graph, label in self._prob_data_generator(p_samples, code, p):
                labels_prob[jdx], graphs_prob[jdx] = label, graph

            graphs_code[idx * p_samples: (idx + 1) * p_samples] = graphs_prob
            labels_code[idx * p_samples: (idx + 1) * p_samples] = labels_prob

        return graphs_code, labels_code

    def _create_ensemble_datasets(self, dir_type, directory):
        split_ratio = self._split_ratio if dir_type == "train" else 1 - self._split_ratio
        dataset_size = int(self._n_samples * split_ratio)
        ens_size = int(dataset_size / len(self._code_distances))

        x_data_ensemble, y_data_ensemble = [], []
        for idx, f in enumerate(glob.glob(str(directory) + f'/*.npz')):
            data = np.load(file=f, allow_pickle=True)
            ens_idx = np.random.choice(dataset_size, size=ens_size, replace=False)
            x_data_ensemble.append(data['graphs'][ens_idx])
            y_data_ensemble.append(data['labels'][ens_idx])

        file_path = directory / f"data_dist_ensemble"
        x_data_ensemble = np.concatenate(x_data_ensemble, axis=0)
        y_data_ensemble = np.concatenate(y_data_ensemble, axis=0)
        np.savez(file_path, graphs=x_data_ensemble, labels=y_data_ensemble)

    def generate_training_data(self):
        data_indexes = list(range(self._n_samples))

        for code_dist in self._code_distances:
            x_data, y_data = self._generate_single_code_data(code_dist)
            train_idx, val_idx = random_split(
                input_list=data_indexes, split_ratio=self._split_ratio)

            print("Splitting and saving datasets...")
            for k, dir_ in self.data_dirs.items():
                split_idx = train_idx if k == "train" else val_idx
                file_path = dir_ / f"data_dist_{code_dist}"
                np.savez(file_path, graphs=x_data[split_idx], labels=y_data[split_idx])

        print(f"Creating code distances ensemble datasets...")
        for k, dir_ in self.data_dirs.items():
            self._create_ensemble_datasets(dir_type=k, directory=dir_)
