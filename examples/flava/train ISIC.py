# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from common.data import MultiDataModule
from flava.callbacks.multimodal_eval import MultimodalEvalCallback
from flava.data import ImageDataModule, MLMDataModule, VLDataModule, ISICDataModule, ISICMLMDataModule, ISICVLDataModule
from flava.definitions import FLAVAArguments
from flava.model import FLAVAPreTrainingLightningModule
from flava.utils import build_config, build_datamodule_kwargs
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch


mim_weights = ["model.model.image_encoder", "model.image_encoder",
               "model.model.image_projection", "model.image_projection",
               "model.image_codebook", "image_codebook"]
mlm_weights = ["model.model.text_encoder", "model.text_encoder",
               "model.model.text_projection", "model.text_projection"]
vl_weights = mim_weights+\
             mlm_weights+\
             ["model.model.mm_encoder", "model.mm_encoder",
              "model.model.image_to_mm_projection", "model.image_to_mm_projection",
              "model.model.text_to_mm_projection", "model.text_to_mm_projection"]

pre_train_2_layers =  {"mim": mim_weights,
                       "mlm": mlm_weights,
                       "vl": vl_weights}

def main():
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    datamodules = []

    # also needed for the imagenet eval callback
    imagenet_datamodule = ISICDataModule(
        **build_datamodule_kwargs(config.datasets.image, config.training),
        type=config.datasets.type
    )
    if "image" in config.datasets.selected:
        datamodules.append(imagenet_datamodule)

    if "text" in config.datasets.selected:
        mlm_datamodule = ISICMLMDataModule(
            **build_datamodule_kwargs(config.datasets.text, config.training),
            type=config.datasets.type,
            tok_type=config.model.text_enc
        )
        datamodules.append(mlm_datamodule)

    if "vl" in config.datasets.selected:
        vl_datamodule = ISICVLDataModule(
            **build_datamodule_kwargs(config.datasets.vl, config.training),
            type=config.datasets.type,
            tok_type=config.model.text_enc
        )
        datamodules.append(vl_datamodule)

    datamodule = MultiDataModule(datamodules)

    datamodule.setup("fit")
    model = FLAVAPreTrainingLightningModule(
        learning_rate=config.training.learning_rate,
        adam_eps=config.training.adam_eps,
        adam_weight_decay=config.training.adam_weight_decay,
        adam_betas=config.training.adam_betas,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.lightning.max_steps,
        random_init_te = config.random_init_te,
        **config.model,
    )

    if "random_init_te" in config and config.random_init_te:
        print("\n\n------------> Random Init Text Encoder <----------------")

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        #MultimodalEvalCallback(imagenet_datamodule=imagenet_datamodule), ravi
        # commented out since for MLM training imagenet dataset is not setup('fit') and causes issues
    ]

    if config.training.lightning_checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                **OmegaConf.to_container(config.training.lightning_checkpoint)
            )
        )
    
    if config.datasets.type == "ISIC":
        val_batches = 254
        test_batches = 0
        train_batches = 25350 // 2
    elif config.datasets.type == "CBIS":
        val_batches = 36
        test_batches = 0
        train_batches = 3600 // 2

    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
        callbacks=callbacks,
        limit_val_batches = val_batches,    # ISIC= 254, CBIS=36
        limit_test_batches = test_batches,
        limit_train_batches = train_batches,  #25*1014=25350(ISIC)/ (CBIS) 25*144=3600,
    )
    ckpt_path = config.training.lightning_load_from_checkpoint
    
    if "pre_init_path" in config and "pre_train_type" in config:
        assert (ckpt_path is None or config.pre_init_path is None)
        sel_dict = {}
        for fpth, pre_train_type in zip(config.pre_init_path, config.pre_train_type):
            pre_train_layers = pre_train_2_layers[pre_train_type]
            w = torch.load(fpth)
        
            if 'state_dict' not in w.keys():    # patching to rectify different naming conventions
                a = w
                w = {}
                w['state_dict'] = a
            for key, val in w['state_dict'].items():
                for sel_tok in pre_train_layers:
                    if sel_tok in key:
                        sel_dict[key] = val
                        break
        _ = model.load_state_dict(sel_dict, strict=False)
        print(f"missing: {_.missing_keys}")
        print(f"unexpected: {_.unexpected_keys}")
    
    if "freeze_te" in config and config.freeze_te:
        print("------> Freezing the text encoder <----------")
        freeze_layers = mlm_weights[:2]
        for name, p in model.named_parameters():
            for freeze_tag in freeze_layers:
                if freeze_tag in name:
                    p.requires_grad = False


    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
