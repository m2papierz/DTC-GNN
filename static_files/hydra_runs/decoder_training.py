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
            filename=callbacks_config.model_filename,
            dirpath=save_path / model_name)
    ]


@hydra.main(
    config_path="../config_files",
    config_name="decoder_training_config",
    version_base=None
)
def main(config: DictConfig):
    hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    data_module = hydra.utils.instantiate(config.data_module)
    gnn_model = hydra.utils.instantiate(config.gnn_model)

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

    lightning_trainer.fit(
        model=gnn_model, datamodule=data_module)
    val_metrics = lightning_trainer.validate(
        model=gnn_model, dataloaders=data_module.val_dataloader())
    test_metrics = lightning_trainer.test(
        model=gnn_model, dataloaders=data_module.test_dataloader())

    print(val_metrics)
    print(test_metrics)


if __name__ == "__main__":
    main()
