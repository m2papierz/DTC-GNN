import sys
import hydra
import random
import os.path
import numpy as np

from tqdm import trange
from typing import List, Tuple, Union
from omegaconf import DictConfig
from pathlib import Path

from qecsim.paulitools import bsp
from dtc_gnn.utlis import init_directory
from qecsim.models.toric import ToricCode
from qecsim.models.generic import DepolarizingErrorModel
from dtc_gnn.data_management.data_transforms import SyndromeToGraphTransform


def get_split_indices(
        data_len: int,
        split_ratio: float,
) -> Tuple[List[int], List[int]]:
    """
    Returns indices split into two subsets with given split ratio.
    """
    assert 0 < split_ratio < 1.0, "split_ratio must be in (0, 1.0)"

    indices = list(range(data_len))
    random.shuffle(indices)
    split_index = int(np.floor(split_ratio * data_len))

    return indices[:split_index], indices[split_index:]


def generate_single_code_data(
        code_dist: int,
        error_probs: List[int],
        n_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data (graph/logical errors pairs) for the single code distance.
    """
    error_model = DepolarizingErrorModel()
    toric_code = ToricCode(rows=code_dist, columns=code_dist)
    to_graph_transform = SyndromeToGraphTransform(code=toric_code)

    x_data, y_data = [], []
    data_size = int(n_samples / len(error_probs))
    for p in error_probs:
        graphs, labels = [], []

        desc = f"Generating data for code_dist={code_dist} and error_prob={p}"
        for _ in trange(data_size, desc=desc, file=sys.stdout):
            errors, syndrome = [], []

            # With no errors/syndrome graph cannot be generated
            while not np.any(errors) or not np.any(syndrome):
                errors = error_model.generate(
                    code=toric_code, probability=p)
                syndrome = bsp(errors, toric_code.stabilizers.T)

            graphs.append(to_graph_transform(syndrome=syndrome))
            labels.append(bsp(errors, toric_code.logicals.T))

        x_data.append(np.array(graphs, dtype=object))
        y_data.append(np.array(labels, dtype=np.float64))

    code_x_data = np.concatenate(x_data, axis=0)
    code_y_data = np.concatenate(y_data, axis=0)

    return code_x_data, code_y_data


def split_and_save_datasets(
        code_data: Tuple[np.ndarray, np.ndarray],
        split_ratio: float,
        ensemble_size: int,
        save_dirs: dir,
        code_dist: Union[int, str]
) -> None:
    """
    Splits data into training and validation subsets.
    """
    x_data, y_data = code_data
    train_ind, val_ind = get_split_indices(
        data_len=len(x_data), split_ratio=split_ratio)

    print("Splitting and saving data...")
    for k, dir_ in save_dirs.items():
        if k == 'train':
            split_ind = train_ind
            ens_size = int(split_ratio * ensemble_size)
        else:
            split_ind = val_ind
            ens_size = int(np.ceil((1.0 - split_ratio) * ensemble_size))
        split_ind_ens = np.random.choice(
            len(split_ind), size=ens_size, replace=False)

        # Saving single code dataset
        file_path = dir_ / f"data_dist_{code_dist}"
        np.savez(file_path, graphs=x_data[split_ind], labels=y_data[split_ind])

        # Saving codes ensemble dataset
        file_path = dir_ / "data_dist_ensemble.npz"
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            graphs = np.concatenate(
                [data['graphs'], x_data[split_ind_ens]], axis=0)
            labels = np.concatenate(
                [data['labels'], y_data[split_ind_ens]], axis=0)
        else:
            graphs = x_data[split_ind_ens]
            labels = y_data[split_ind_ens]
        np.savez(str(file_path).rstrip('.npz'), graphs=graphs, labels=labels)


@hydra.main(
    config_path="../config_files",
    config_name="data_generation_config",
    version_base=None
)
def main(config: DictConfig):
    save_dirs = {k: Path(sys.path[1]) / v for k, v in config.save_dirs.items()}
    for dir_ in save_dirs.values():
        init_directory(dir_)

    # Size of the code distances ensemble dataset
    ensemble_size = int(config.n_samples / len(config.code_distances))

    # Create dataset for as single code distance
    for code_dist in config.code_distances:
        data = generate_single_code_data(
            code_dist=code_dist,
            error_probs=config.error_probs,
            n_samples=config.n_samples
        )

        split_and_save_datasets(
            code_data=data,
            split_ratio=config.split_ratio,
            ensemble_size=ensemble_size,
            code_dist=code_dist,
            save_dirs=save_dirs
        )


if __name__ == "__main__":
    main()
