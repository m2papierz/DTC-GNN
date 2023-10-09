import hydra
import path_init

from omegaconf import DictConfig
from dtc_gnn.utlis import data_dirs_status
from dtc_gnn.data_generation.generator import GraphDataGenerator


@hydra.main(
    config_path="../config_files",
    config_name="data_generation_config",
    version_base=None
)
def main(config: DictConfig):
    generator: GraphDataGenerator = hydra.utils.instantiate(config.data_generator)

    if data_dirs_status(dir_paths=list(generator.data_dirs.values())):
        generator.generate_training_data()


if __name__ == "__main__":
    main()
