import sys
import hydra
import path_init
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig
from dtc_gnn.simulation.decoding_simulator import Simulator
from dtc_gnn.simulation.decoders.mwpm_decoder import DecoderMWPM
from dtc_gnn.simulation.decoders.gnn_decoder import DecoderGNN

from dtc_gnn.deep_decoding.lightning_module import GNNLightningModule


def plot_errors(log_errors, code_distances, prob_linspace, num_shots):
    for L, logical_errors in zip(code_distances, log_errors):
        std_err = (logical_errors * (1 - logical_errors) / num_shots) ** 0.5
        plt.errorbar(prob_linspace, logical_errors, yerr=std_err, label="L={}".format(L))
    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    plt.legend(loc=0)
    plt.show()


@hydra.main(
    config_path="../config_files",
    config_name="decoder_simulation_config",
    version_base=None
)
def main(config: DictConfig):
    simulator: Simulator = hydra.utils.instantiate(config.simulator)
    gnn_model: GNNLightningModule = hydra.utils.instantiate(config.gnn_model)

    mwpm_decoder = DecoderMWPM()
    gnn_decoder = DecoderGNN(
        model_path=Path(sys.path[1]) / config.gnn_model_path,
        torch_module=gnn_model
    )

    for decoder in [gnn_decoder]:
        results = simulator.run_simulation(decoder=decoder)
        plot_errors(**results)


if __name__ == "__main__":
    main()
