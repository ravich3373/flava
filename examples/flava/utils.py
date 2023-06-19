# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from flava.definitions import (
    FLAVAArguments,
    TrainingArguments,
    TrainingSingleDatasetInfo,
)
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_datamodule_kwargs(
    dm_config: TrainingSingleDatasetInfo, training_config: TrainingArguments
):
    if dm_config.test:
        kwargs = {
        "train_infos": dm_config.train,
        "val_infos": dm_config.val,
        "test_infos": dm_config.test,
        "batch_size": dm_config.batch_size or training_config.batch_size,
        "num_workers": dm_config.num_workers or training_config.num_workers,
        "allow_uneven_batches": dm_config.allow_uneven_batches,
        }
    else:
        kwargs = {
            "train_infos": dm_config.train,
            "val_infos": dm_config.val,
            "batch_size": dm_config.batch_size or training_config.batch_size,
            "num_workers": dm_config.num_workers or training_config.num_workers,
            "allow_uneven_batches": dm_config.allow_uneven_batches,
        }
    kwargs.update(dm_config.datamodule_extra_kwargs)
    return kwargs


def build_config():
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError(
            "Please pass 'config' to specify configuration yaml file for running FLAVA"
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = instantiate(yaml_conf)
    cli_conf.pop("config")
    config: FLAVAArguments = OmegaConf.merge(conf, cli_conf)

    assert (
        "max_steps" in config.training.lightning
    ), "lightning config must specify 'max_steps'"

    return config
