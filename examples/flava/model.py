# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, Recall, ConfusionMatrix, MulticlassAUROC
from torchmultimodal.models.flava.model import (
    flava_model_for_classification,
    flava_model_for_pretraining,
)
from transformers.optimization import get_cosine_schedule_with_warmup


def get_optimizers_for_lightning(
    model: torch.nn.Module,
    learning_rate: float,
    adam_eps: float,
    adam_weight_decay: float,
    adam_betas: Tuple[float, float],
    warmup_steps: int,
    max_steps: int,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=adam_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class FLAVAPreTrainingLightningModule(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0002,
        adam_eps: float = 1.0e-08,
        adam_weight_decay: float = 0.01,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 2000,
        max_steps: int = 450000,
        **flava_pretraining_kwargs: Any,
    ):
        super().__init__()
        self.model = flava_model_for_pretraining(image_size=384,codebook_image_size=192, **flava_pretraining_kwargs)
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_betas = adam_betas
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        losses = output.losses
        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                self.log(f"train/losses/{key}", losses[key], prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        losses = output.losses
        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                self.log(
                    f"validation/losses/{key}", losses[key], prog_bar=True, logger=True
                )

        return total_loss

    def _step(self, batch, batch_idx):
        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")

        output = self.model(
            image=batch.get("image", None),
            image_for_codebook=batch.get("image_for_codebook", None),
            image_patches_mask=batch.get("image_patches_mask", None),
            text=batch.get("text", None),
            text_masked=batch.get("text_masked", None),
            mlm_labels=batch.get("mlm_labels", None),
            itm_labels=batch.get("itm_labels", None),
            required_embedding=required_embedding,
        )
        return output

    def configure_optimizers(self):
        return get_optimizers_for_lightning(
            self.model,
            self.learning_rate,
            self.adam_eps,
            self.adam_weight_decay,
            self.adam_betas,
            self.warmup_steps,
            self.max_steps,
        )

# ravi - torch metrics
# https://github.com/Lightning-AI/lightning/issues/4396
# https://github.com/Lightning-AI/torchmetrics/issues/789
class FLAVAClassificationLightningModule(LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 0.0002,
        adam_eps: float = 1.0e-08,
        adam_weight_decay: float = 0.01,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 2000,
        max_steps: int = 450000,
        ds_type = "ISIC",
        **flava_classification_kwargs: Any,
    ):
        super().__init__()
        self.model = flava_model_for_classification(
            num_classes, image_size=384, **flava_classification_kwargs
        )
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.adam_betas = adam_betas
        if ds_type == "ISIC":
            self.acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
            self.re = Recall(task="multiclass", num_classes=num_classes, average=None)
            self.avg_re = Recall(task="multiclass", num_classes=num_classes, average="macro")

            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
            self.val_re = Recall(task="multiclass", num_classes=num_classes, average=None)
            self.val_avg_re = Recall(task="multiclass", num_classes=num_classes, average="macro")
        elif ds_type == "CBIS":
            self.acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
            self.re = Recall(task="multiclass", num_classes=num_classes, average=None)
            self.avg_re = MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=None)

            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
            self.val_re = Recall(task="multiclass", num_classes=num_classes, average=None)
            self.val_avg_re = MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=None)


    def log_by_class(self, metric_tensor, matric_name, split):
        for cls_idx, metric in enumerate(metric_tensor):
            self.log(
            f"{split}/{matric_name}_cls_{cls_idx}/classification",
            metric,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_step=False, on_epoch=True
            )

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx, train=True)
        self.log("train/losses/classification", output.loss, prog_bar=True, logger=True)
        
        self.log(
            "train/accuracy/classification",
            self.acc,
            sync_dist=True,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True
        )

        self.log(
            "train/micro_avg_recall/classification",
            self.avg_re,
            sync_dist=True,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True
        )


        #self.log_by_class(re, "recall", "train")

        return output.loss

    def test_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx, train=False)
        self.log("test/losses/classification", output.loss, prog_bar=True, logger=True)
        
        self.log(
            "test/accuracy/classification",
            self.val_acc,
            sync_dist=True,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True
        )

        self.log(
            "test/micro_avg_recall/classification",
            self.val_avg_re,
            sync_dist=True,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True
        )
        #self.log_by_class(re, "recall", "train")

        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx, train=False)
        self.log(
            "validation/losses/classification", output.loss, prog_bar=True, logger=True
        )
    
        #self.log_by_class(pre, "recall", "validation")

        return output.loss

    def _step(self, batch, batch_idx, train=False):
        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")

        labels = batch["labels"]
        output = self.model(
            image=batch.get("image", None),
            text=batch.get("text", None),
            required_embedding=required_embedding,
            labels=labels,
        )
        
        if train:
            accuracy = self.acc.update(output.logits, labels)
            re = self.re.update(output.logits, labels)
            avg_re = self.avg_re.update(output.logits, labels)
        else:
            accuracy = self.val_acc.update(output.logits, labels)
            re = self.val_re.update(output.logits, labels)
            avg_re = self.val_avg_re.update(output.logits, labels)

        return output

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        avg_re = self.val_avg_re.compute()
        re = self.val_re.compute()

        self.log(
            "validation/accuracy/classification",
            acc,
            sync_dist=True,
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True
        )
        self.log(
            "validation/micro_avg_recall/classification",
            avg_re,
            sync_dist=True,
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True
        )

        self.log_by_class(re, "recall", "validation")



    def on_test_end_(self):
        acc = self.val_acc.compute()
        avg_re = self.val_avg_re.compute()
        re = self.val_re.compute()

        self.log(
            "validation/accuracy/classification",
            acc,
            sync_dist=True,
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True
        )
        self.log(
            "validation/micro_avg_recall/classification",
            avg_re,
            sync_dist=True,
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True)

        self.log_by_class(re, "recall", "test")



    def configure_optimizers(self):
        return get_optimizers_for_lightning(
            self.model,
            self.learning_rate,
            self.adam_eps,
            self.adam_weight_decay,
            self.adam_betas,
            self.warmup_steps,
            self.max_steps,
        )
