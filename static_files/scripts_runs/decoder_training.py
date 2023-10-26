import hydra
import torch
import path_init
import pytorch_lightning.callbacks

from typing import List
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from dtc_gnn.data_management.modules.graph_data_module import GraphDataModule
from dtc_gnn.deep_decoding.lightning_module import GNNLightningModule

torch.set_float32_matmul_precision('medium')


def get_callbacks(
        callbacks_config: DictConfig,
        save_path: Path,
        model_name: str
) -> List[pytorch_lightning.callbacks.Callback]:
    return [
        EarlyStopping(
            monitor=callbacks_config.monitor,
            min_delta=callbacks_config.min_delta,
            patience=callbacks_config.patience,
            mode=callbacks_config.mode,
            verbose=callbacks_config.verbose),
        ModelCheckpoint(
            monitor=callbacks_config.monitor,
            save_top_k=callbacks_config.save_top_k,
            mode=callbacks_config.mode,
            filename=model_name,
            dirpath=save_path)
    ]


@hydra.main(
    config_path="../config_files",
    config_name="decoder_training_config",
    version_base=None
)
def main(config: DictConfig):
    hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    data_module: GraphDataModule = hydra.utils.instantiate(config.data_module)
    gnn_model: GNNLightningModule = hydra.utils.instantiate(config.gnn_model)

    trainer_callbacks = get_callbacks(
        callbacks_config=config.trainer.callbacks,
        save_path=hydra_path,
        model_name=gnn_model.__class__.__name__
    )

    lightning_trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=trainer_callbacks,
        default_root_dir=hydra_path
    )

    lightning_trainer.fit(model=gnn_model, datamodule=data_module)


if __name__ == "__main__":
    main()
