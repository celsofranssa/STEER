import os

import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer

from source.datamodule.SiameseDataModule import SiameseDataModule
from source.model.SiameseModel import SiameseModel


def get_logger(params):
    return loggers.TensorBoardLogger(
        save_dir=params.log.dir,
        name=f"{params.model.name}_{params.data.name}_exp"
    )


def get_model_checkpoint_callback(params):
    return ModelCheckpoint(
        monitor="val_Cos-SMLTY",
        dirpath=params.model_checkpoint.dir,
        filename=f"{params.model.name}_{params.data.name}",
        save_top_k=1,
        save_weights_only=True,
        mode="max"
    )


def get_early_stopping_callback(params):
    return EarlyStopping(
        monitor='val_Cos-SMLTY',
        patience=params.trainer.patience,
        min_delta=params.trainer.min_delta,
        mode='max'
    )


def get_tokenizer(hparams):
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.tokenizer.architecture
    )
    if hparams.tokenizer.architecture == "gpt2":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


class SiameseDataModel:
    pass


def fit(params):


    print(f"Fitting {params.model.name} over {params.data.name} with fowling params\n"
          f"{OmegaConf.to_yaml(params)}\n")

    # Initialize a trainer
    trainer = pl.Trainer(
        fast_dev_run=params.trainer.fast_dev_run,
        max_epochs=params.trainer.max_epochs,
        precision=params.trainer.precision,
        gpus=params.trainer.gpus,
        progress_bar_refresh_rate=params.trainer.progress_bar_refresh_rate,
        logger=get_logger(params),
        callbacks=[
            get_model_checkpoint_callback(params),  # checkpoint_callback
            get_early_stopping_callback(params),  # early_stopping_callback
        ]
    )
    # Train the ⚡ model
    trainer.fit(
        model=SiameseModel(params.model),
        datamodule=SiameseDataModule(params.data, get_tokenizer(params.model))
    )


def test(params):

    for fold in params.data.folds:
        print(f"Predicting {params.model.name} over {params.data.name} (fold {fold}) with fowling params\n"
              f"{OmegaConf.to_yaml(params)}\n")

        # load model checkpoint
        model = SiameseModel.load_from_checkpoint(
            checkpoint_path=f"{params.model_checkpoint.dir}{params.model.name}_{params.data.name}_{fold}.ckpt"
        )

        model.hparams.stat.name = f"{params.model.name}_{params.data.name}_{fold}.stat"

        # trainer
        trainer = pl.Trainer(
            gpus=params.trainer.gpus
        )

        # testing
        trainer.test(
            model=model,
            datamodule=SiameseDataModule(params.data, get_tokenizer(params.model))
        )



@hydra.main(config_path="settings/", config_name="settings.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)
    if "fit" in params.tasks:
        fit(params)
    if "test" in params.tasks:
        test(params)


if __name__ == '__main__':
    perform_tasks()
