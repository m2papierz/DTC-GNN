import sys
import hydra
import random
import os.path
import path_init
import numpy as np
import qecsim.paulitools as pt

from tqdm import trange
from typing import List, Tuple, Union
from omegaconf import DictConfig
from pathlib import Path

from dtc_gnn.utlis import data_dirs_status
from qecsim.models.toric import ToricCode


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


def generate_single_code_data(
        code_dist: int,
        error_prob: float,
        n_samples: int,
        graph_transform: any,
        error_model: any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data (graph/logical errors pairs) for the single code distance.
    """
    desc = f"Generating data for code_dist={code_dist}"
    toric_code = ToricCode(rows=code_dist, columns=code_dist)

    graphs = np.empty(shape=n_samples, dtype=object)
    labels = np.empty(shape=n_samples, dtype=object)
    for idx in trange(n_samples, file=sys.stdout, desc=desc):
        errors, syndrome = [], []
        while not np.any(errors) or not np.any(syndrome):
            errors = error_model.generate(
                shape=toric_code.n_k_d[0], probability=error_prob)
            syndrome = pt.bsp(errors, toric_code.stabilizers.T)

        graph = graph_transform(stab_code=toric_code, syndrome=syndrome)
        logical_error = pt.bsf_to_pauli(pt.bsp(errors, toric_code.logicals.T))
        labels[idx], graphs[idx] = logical_error, graph

    return graphs, labels


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
    data_indexes = list(range(len(x_data)))
    train_idx, val_idx = random_split(
        input_list=data_indexes, split_ratio=split_ratio)
    val_idx, test_idx = random_split(
        input_list=val_idx, split_ratio=0.5)

    ens_size_train = int(split_ratio * ensemble_size)
    ens_size_val = int(np.ceil((1.0 - split_ratio) * ensemble_size))
    split_dir = {
        "train": [train_idx, ens_size_train],
        "val": [val_idx, ens_size_val],
        "test": [test_idx, ens_size_val]
    }

    print("Splitting and saving datasets...")
    for k, dir_ in save_dirs.items():
        split_idx, ens_size = split_dir[k]
        split_idx_ens = np.random.choice(
            len(split_idx), size=ens_size, replace=False)

        # Saving single code dataset
        file_path = dir_ / f"data_dist_{code_dist}"
        np.savez(file_path, graphs=x_data[split_idx], labels=y_data[split_idx])

        # Saving codes ensemble dataset
        file_path = dir_ / "data_dist_ensemble.npz"
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            graphs = np.concatenate(
                [data['graphs'], x_data[split_idx_ens]], axis=0)
            labels = np.concatenate(
                [data['labels'], y_data[split_idx_ens]], axis=0)
        else:
            graphs = x_data[split_idx_ens]
            labels = y_data[split_idx_ens]
        np.savez(str(file_path).rstrip('.npz'), graphs=graphs, labels=labels)


@hydra.main(
    config_path="../config_files",
    config_name="data_generation_config",
    version_base=None
)
def main(config: DictConfig):
    error_model = hydra.utils.instantiate(config.error_model)
    to_graph_transform = hydra.utils.instantiate(config.graph_transform)

    data_dirs = {k: Path(sys.path[1]) / v for k, v in config.data_dirs.items()}
    if data_dirs_status(dir_paths=list(data_dirs.values())):
        # Size of the code distances ensemble dataset
        ensemble_size = int(config.n_samples / len(config.code_distances))

        # Create dataset for as single code distance
        for code_dist in config.code_distances:
            data = generate_single_code_data(
                code_dist=code_dist,
                error_prob=config.error_prob,
                n_samples=config.n_samples,
                error_model=error_model,
                graph_transform=to_graph_transform
            )

            split_and_save_datasets(
                code_data=data,
                split_ratio=config.split_ratio,
                ensemble_size=ensemble_size,
                code_dist=code_dist,
                save_dirs=data_dirs
            )


if __name__ == "__main__":
    main()
