import sys
import hydra
import path_init
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig
from typing import List

from dtc_gnn.simulation.decoding_simulator import Simulator
from dtc_gnn.simulation.decoders.mwpm_decoder import DecoderMWPM
from dtc_gnn.simulation.decoders.gnn_decoder import DecoderGNN

from dtc_gnn.deep_decoding.lightning_module import GNNLightningModule


def get_results_save_path(
        config: DictConfig,
        s="_"
) -> Path:
    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(exist_ok=True)
    date_time = s.join(config.gnn_model_path.split("/")[1:3])
    return reports_dir / date_time


def plot_and_save_results(
        results_dict: dict,
        code_dist: List[int],
        max_error_prob: float,
        prob_space_num: int,
        n_shots: int,
        save_dir: Path,
        plt_colors: List[str] = None
) -> None:
    if plt_colors is None:
        plt_colors = ['tab:orange', 'tab:blue', 'tab:green']

    prob_linspace = np.linspace(
        start=0.0, stop=max_error_prob,
        num=prob_space_num)

    plt.figure()
    for decoder_name, log_e in results_dict.items():
        line_style = '--' if decoder_name == 'MWPM' else '-'

        for i, (L, logical_errors) in enumerate(zip(code_dist, log_e)):
            std_err = (logical_errors * (1 - logical_errors) / n_shots) ** 0.5
            plt.errorbar(
                x=prob_linspace, y=logical_errors, yerr=std_err,
                label=f"{decoder_name} - L={L}", ls=line_style,
                color=plt_colors[i % 3]
            )

    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    plt.legend(loc=0)

    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "results_plot")


@hydra.main(
    config_path="../config_files",
    config_name="decoder_simulation_config",
    version_base=None
)
def main(config: DictConfig):
    simulator_config = config.simulator
    simulator: Simulator = hydra.utils.instantiate(config.simulator)
    gnn_model: GNNLightningModule = hydra.utils.instantiate(config.gnn_model)

    # Instantiate decoders
    mwpm_decoder = DecoderMWPM()
    gnn_decoder = DecoderGNN(
        model_path=Path(sys.path[1]) / config.gnn_model_path,
        torch_module=gnn_model
    )

    # Collect results fo decoding simulation
    decoding_results = {}
    for decoder in [gnn_decoder, mwpm_decoder]:
        print(f"Simulating of {decoder.name} decoder...")
        decoding_results.update(
            {decoder.name: simulator.run_simulation(decoder=decoder)}
        )

    plot_and_save_results(
        results_dict=decoding_results,
        code_dist=simulator_config.code_distances,
        max_error_prob=simulator_config.max_error_prob,
        prob_space_num=simulator_config.prob_space_num,
        n_shots=simulator_config.num_shots,
        save_dir=get_results_save_path(config)
    )


if __name__ == "__main__":
    main()
