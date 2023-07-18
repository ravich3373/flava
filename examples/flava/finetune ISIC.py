# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from flava.data import TextDataModule, TorchVisionDataModule, ISICTorchVisionDataModule
from flava.data.datamodules import ISICVLDataModule
from flava.definitions import FLAVAArguments
from flava.model import FLAVAClassificationLightningModule
from flava.utils import build_config, build_datamodule_kwargs
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

AVAIL_GPUS = 1
SEED = -1
NUM_CLASSES = 2

NUM_WORKERS = 4
MAX_STEPS = 24000
BATCH_SIZE = 32


exclude = ["image_codebook", "model.loss"]


def main():
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    assert len(config.datasets.selected) == 1
    if "image" in config.datasets.selected:
        datamodule = ISICTorchVisionDataModule(
            **build_datamodule_kwargs(config.datasets.image, config.training)
        )
    # elif "text":
    #     datamodule = TextDataModule(
    #         **build_datamodule_kwargs(config.datasets.text, config.training)
    #     )
    else:
        datamodule = ISICVLDataModule(
            **build_datamodule_kwargs(config.datasets.vl, config.training),
            finetuning=True,
            itm_probability=-1,  # ravi dont want to mismatch images and texts.
            use_dict=True, unnest=True
        )

    datamodule.setup("fit")

    model = FLAVAClassificationLightningModule(
        num_classes=config.datasets.num_classes,
        learning_rate=config.training.learning_rate,
        adam_eps=config.training.adam_eps,
        adam_weight_decay=config.training.adam_weight_decay,
        adam_betas=config.training.adam_betas,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.lightning.max_steps,
        **config.model,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]

    if config.training.lightning_checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                **OmegaConf.to_container(config.training.lightning_checkpoint)
            )
        )

    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning), callbacks=callbacks,
        limit_val_batches = 254,
        limit_test_batches=317,
        limit_train_batches=25350,  #25*1014,
        max_epochs=15
    )
    ckpt_path = config.training.lightning_load_from_checkpoint

    if "init_path" in config:
        assert (ckpt_path is None or config.init_path is None)
        w = torch.load(config.init_path)
        sel_w = {}
        for key, val in w["state_dict"].items():
            sel = True
            for esc_key in exclude:
                if esc_key in key:
                    sel = False
            if sel:
                sel_w[key] = val

        _ = model.load_state_dict(sel_w, strict=False)
        print(f"missing: {_.missing_keys}")
        print(f"unexpected: {_.unexpected_keys}")
    
    if "linear_probe" in config:
        if config.linear_probe:
            # free all the layers other than model.model.classifier
            for name, p in model.named_parameters():
                if not "model.classifier" in name:
                    p.requires_grad = False
                else:
                    print(f"Training weight: {name}")

    #trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    #trainer.validate(datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
