# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

import torch
import torchvision
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
from flava.definitions import HFDatasetInfo, TorchVisionDatasetInfo, ISICInfo
from pytorch_lightning import LightningDataModule
from transformers import (
    BertTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DefaultDataCollator,
    TRANSFORMERS_CACHE,
)
from transformers.data.data_collator import torch_default_data_collator

from .transforms import (
    default_image_pretraining_transforms,
    default_text_transform,
    default_torchvision_transforms,
    encode_text_batch,
    pad_batch,
    TEXT_DEFAULT_TOKENIZER,
    TEXT_WHOLE_WORD_MASK_TOKENIZER,
    VL_MAX_LENGTH_DEFAULT,
    VLTransform,
)
from .utils import build_datasets_from_info, fetch_images


def transform_image(transform, sample):
    sample.update(transform(sample["img"]))# image ravi
    sample.pop("img")
    return sample


class ISICDataset(Dataset):
    def __init__(self,
                 csv_path: Union[Path, str],
                 img_path: Union[Path, str],
                 mode):
        super().__init__()
        self.cols = ["image_name", "diagnosis", "description"]
        self.df = pd.read_csv(csv_path, usecols=self.cols)
        self.img_dir = img_path
        self.transforms = None
        self.mode = mode
        self.label_num2str = {0: 'NV',
                              1: 'MEL',
                              2: 'BCC',
                              3: 'BKL',
                              4: 'AK',
                              5: 'SCC',
                              6: 'VASC',
                              7: 'DF'}
        self.label_str2num = {'NV': 0,
                              'MEL':1,
                              'BCC':2,
                              'BKL':3,
                              'AK':4,
                              'SCC':5,
                              'VASC':6,
                              'DF':7}
        self.df['diagnosis'] = self.df["diagnosis"].apply(lambda l: self.label_str2num[l])
        if mode == "text":
            self.df = self.df[["diagnosis", "description"]]
        elif mode == "image":
            self.df = self.df[["diagnosis", "image_name"]]
        elif mode == "mlm":
            self.df = self.df[["description"]]
        
        d = {"image_name": "image",
             "diagnosis": "label",
             "description": "text"}
        self.df.rename(mapper=d, axis=1, inplace=True)

    def __getitem__(self, index):
        data = self.df.iloc[index].to_dict()
        if "image" in data.keys():
            im_pth = os.path.join(self.img_dir, data['image'])
            img = Image.open(im_pth)
            data['image'] = img

        if self.mode == "image" and self.transforms is not None:
            data.update(self.transforms(img))                
        elif self.mode == "mlm" and self.transforms is not None:
            return self.transforms(data)
        elif self.mode == "vl" and self.transforms is not None:
            return self.transforms(data)
        return data

    def __len__(self):
        return len(self.df)

    def set_transform(self, transforms):
        self.transforms = transforms

    def select(self, idxs):
        idx = idxs[0]
        data = self.df.iloc[idx].to_dict()
        return [data]


class DataCollatorForWholeWordMaskRetainingBatch(DataCollatorForWholeWordMask):
    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        masked_batch = super().torch_call(examples)
        examples = torch_default_data_collator(examples)
        examples["input_ids"] = masked_batch["input_ids"]
        examples["labels"] = masked_batch["labels"]
        return examples


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        train_infos: List[HFDatasetInfo],
        val_infos: Optional[List[HFDatasetInfo]] = None,
        transforms: Optional[Tuple[Callable, Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        allow_uneven_batches: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.train_dataset_infos = train_infos
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_uneven_batches = allow_uneven_batches

        if transforms is None:
            transforms = default_image_pretraining_transforms()

        self.train_transform, self.test_transform = transforms

    def setup(self, stage=None):
        train_transform = partial(transform_image, self.train_transform)
        val_transform = partial(transform_image, self.test_transform)
        
        self.train_dataset = build_datasets_from_info(
            self.train_dataset_infos, split="train"
        )
        self.train_dataset.set_transform(train_transform)
        self.val_dataset = build_datasets_from_info(
            self.val_dataset_infos, split="test"#validation ravi
        )
        self.val_dataset.set_transform(val_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=True,
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            # ideally, we don't need to drop these for unimodal cases
            # but just to be safe
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=False,
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            # ideally, we don't need to drop these for unimodal cases
            # but just to be safe
            drop_last=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def on_before_batch_transfer(self, batch, *args):
        if batch["label"].size(0) < self.batch_size and not self.allow_uneven_batches:
            batch = pad_batch(batch, self.batch_size)
        return batch


class TextDataModule(LightningDataModule):
    def __init__(
        self,
        train_infos: List[HFDatasetInfo],
        text_columns: List[str],
        val_infos: Optional[List[HFDatasetInfo]] = None,
        tokenizer: Optional[Callable] = None,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        allow_uneven_batches: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.train_dataset_infos = train_infos
        self.text_columns = text_columns
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_uneven_batches = allow_uneven_batches

    def setup(self, stage=None):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(TEXT_DEFAULT_TOKENIZER)
        transform = partial(
            encode_text_batch,
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            text_columns=self.text_columns,
            return_batch=True,
        )
        self.train_dataset = build_datasets_from_info(
            self.train_dataset_infos, split="train"
        )
        self.train_dataset.set_transform(transform)
        self.val_dataset = build_datasets_from_info(
            self.val_dataset_infos, split="validation"
        )
        self.val_dataset.set_transform(transform)

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def _build_dataloader(self, dataset, drop_last=False, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=shuffle,
            collate_fn=self._build_collator(),
            drop_last=drop_last,
        )

    def _build_collator(self):
        return DefaultDataCollator()

    def on_before_batch_transfer(self, batch, *args):
        batch.pop("token_type_ids", None)
        mask = batch.pop("attention_mask", None)
        if mask.size(0) < self.batch_size and not self.allow_uneven_batches:
            batch = pad_batch(batch, self.batch_size)
        return batch

    def on_after_batch_transfer(self, batch, *args):
        batch["text"] = batch.pop("input_ids")
        return batch


class MLMDataModule(TextDataModule):
    def __init__(
        self,
        train_infos: List[HFDatasetInfo],
        text_columns: List[str],
        val_infos: Optional[List[HFDatasetInfo]] = None,
        mlm_probability: float = 0.15,
        ignore_index: int = -1,
        **kwargs: Any,
    ):
        super().__init__(train_infos, text_columns, val_infos, **kwargs)
        self.mlm_probability = mlm_probability
        self.ignore_index = ignore_index

    def setup(self, stage=None):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(TEXT_DEFAULT_TOKENIZER)
        transform = partial(
            encode_text_batch,
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            text_columns=self.text_columns,
            return_batch=False,
        )
        self.train_dataset = build_datasets_from_info(
            self.train_dataset_infos, split="train"
        )
        self.train_dataset.set_transform(transform)
        self.val_dataset = build_datasets_from_info(
            self.val_dataset_infos, split="validation"
        )
        self.val_dataset.set_transform(transform)

    def _build_dataloader(self, dataset, drop_last=True, shuffle=True):
        # uneven batches can cause distributed issues,
        # drop last batch to prevent those.
        # ideally, we don't need to drop these for unimodal cases
        # but just to be safe
        return super()._build_dataloader(dataset, drop_last=drop_last, shuffle=shuffle)

    def _build_collator(self):
        return DataCollatorForLanguageModeling(
            self.tokenizer, mlm_probability=self.mlm_probability
        )

    def on_after_batch_transfer(self, batch, *args):
        batch["text_masked"] = batch.pop("input_ids")
        batch["mlm_labels"] = batch.pop("labels")
        batch["mlm_labels"][batch["mlm_labels"] == -100] = self.ignore_index
        return batch


class ISICMLMDataModule(TextDataModule):
    def __init__(
        self,
        train_infos: List[HFDatasetInfo],
        text_columns: List[str] = ["text"],
        val_infos: Optional[List[HFDatasetInfo]] = None,
        mlm_probability: float = 0.15,
        ignore_index: int = -1,
        **kwargs: Any,
    ):
        super().__init__(train_infos, text_columns, val_infos, **kwargs)
        self.mlm_probability = mlm_probability
        self.ignore_index = ignore_index

    def setup(self, stage=None):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(TEXT_DEFAULT_TOKENIZER)
        transform = partial(
            encode_text_batch,
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            text_columns=self.text_columns,
            return_batch=False,
        )
        self.train_dataset = ISICDataset(self.train_dataset_infos[0]["csv_path"],
                                         self.train_dataset_infos[0]["img_path"],
                                         "mlm")
        self.train_dataset.set_transform(transform)
        self.val_dataset = ISICDataset(self.val_dataset_infos[0]["csv_path"],
                                       self.val_dataset_infos[0]["img_path"],
                                       "mlm")
        self.val_dataset.set_transform(transform)

    def _build_dataloader(self, dataset, drop_last=True, shuffle=True):
        # uneven batches can cause distributed issues,
        # drop last batch to prevent those.
        # ideally, we don't need to drop these for unimodal cases
        # but just to be safe
        return super()._build_dataloader(dataset, drop_last=drop_last, shuffle=shuffle)

    def _build_collator(self):
        return DataCollatorForLanguageModeling(
            self.tokenizer, mlm_probability=self.mlm_probability
        )

    def on_after_batch_transfer(self, batch, *args):
        batch["text_masked"] = batch.pop("input_ids")
        batch["mlm_labels"] = batch.pop("labels")
        batch["mlm_labels"][batch["mlm_labels"] == -100] = self.ignore_index
        return batch


class VLDataModule(LightningDataModule):
    def __init__(
        self,
        train_infos: List[HFDatasetInfo],
        val_infos: List[HFDatasetInfo],
        text_transform: Optional[Callable] = None,
        image_transforms: Optional[Tuple[Callable, Callable]] = None,
        mlm_probablity: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        finetuning: bool = False,
        ignore_index: int = -1,
        itm_probability: float = 0.1,
        allow_uneven_batches: bool = False,
        fetch_num_threads: int = 4,
        fetch_retries: int = 0,
        fetch_sleep_timer: int = 0,
        fetch_timeout: Optional[float] = None,
        fetch_batch_size: int = 50,
        **kwargs,
    ):
        super().__init__()

        self.train_dataset_infos = train_infos
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos
        if image_transforms is None:
            if not finetuning:
                image_transforms = default_image_pretraining_transforms()
            else:
                image_transforms = default_torchvision_transforms(use_dict=True)

        self.train_image_transform, self.test_image_transform = image_transforms
        self.text_transform = text_transform
        self.mlm_probability = mlm_probablity
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_index = ignore_index
        self.itm_probability = itm_probability
        self.allow_uneven_batches = allow_uneven_batches
        self.fetch_num_threads = fetch_num_threads
        self.fetch_retries = fetch_retries
        self.fetch_sleep_timer = fetch_sleep_timer
        self.fetch_timeout = fetch_timeout
        self.fetch_batch_size = fetch_batch_size

    def setup(self, stage=None):
        if self.text_transform is None:
            # TODO Update to use whole word mask vocab
            text_tokenizer = BertTokenizer.from_pretrained(
                TEXT_WHOLE_WORD_MASK_TOKENIZER
            )
            self.text_transform = default_text_transform(
                text_tokenizer, max_text_length=VL_MAX_LENGTH_DEFAULT
            )
        self.text_tokenizer = self.text_transform.keywords["tokenizer"]
        train_vl_transform = VLTransform(
            self.train_image_transform, self.text_transform
        )
        val_vl_transform = VLTransform(self.test_image_transform, self.text_transform)

        train_dataset = build_datasets_from_info(
            self.train_dataset_infos, split="train"
        )
        train_dataset = train_dataset.map(
            fetch_images,
            batched=True,
            batch_size=self.fetch_batch_size,
            fn_kwargs={
                "num_threads": self.fetch_num_threads,
                "timeout": self.fetch_timeout,
                "retries": self.fetch_retries,
                "sleep_timer": self.fetch_sleep_timer,
            },
        )
        train_dataset = train_dataset.filter(
            lambda example: example["image"] is not None
        )
        self.train_dataset = train_dataset
        self.train_dataset.set_transform(
            partial(
                train_vl_transform,
                dataset=train_dataset.filter(lambda example: True),
                itm_probability=self.itm_probability,
            )
        )

        val_dataset = build_datasets_from_info(
            self.val_dataset_infos, split="validation"
        )

        val_dataset = val_dataset.map(
            fetch_images,
            batched=True,
            batch_size=self.fetch_batch_size,
            fn_kwargs={
                "num_threads": self.fetch_num_threads,
                "timeout": self.fetch_timeout,
                "retries": self.fetch_retries,
                "sleep_timer": self.fetch_sleep_timer,
            },
        )
        val_dataset = val_dataset.filter(lambda example: example["image"] is not None)
        self.val_dataset = val_dataset
        self.val_dataset.set_transform(
            partial(
                val_vl_transform,
                dataset=self.val_dataset.filter(
                    lambda example: True
                ),  # Pass a copy to transform
                itm_probability=self.itm_probability,
            )
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=True,
            collate_fn=self._build_collator(),
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=False,
            collate_fn=self._build_collator(),
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            drop_last=True,
        )

    def _build_collator(self):
        return DataCollatorForWholeWordMaskRetainingBatch(
            self.text_tokenizer, mlm_probability=self.mlm_probability
        )

    def on_before_batch_transfer(self, batch, *args):
        batch.pop("token_type_ids", None)
        mask = batch.pop("attention_mask", None)
        if (
            mask is not None
            and mask.size(0) < self.batch_size
            and not self.allow_uneven_batches
        ):
            batch = pad_batch(batch, self.batch_size)
        return batch

    def on_after_batch_transfer(self, batch, *args):
        text_masked = batch.pop("input_ids")
        mlm_labels = batch.pop("labels", None)
        mlm_labels[mlm_labels == -100] = self.ignore_index
        text = text_masked.detach().clone()
        text[mlm_labels != -1] = mlm_labels[mlm_labels != -1]
        batch.update(
            {"mlm_labels": mlm_labels, "text": text, "text_masked": text_masked}
        )
        return batch


class ISICVLDataModule(LightningDataModule):
    def __init__(
        self,
        train_infos: List[HFDatasetInfo],
        val_infos: List[HFDatasetInfo],
        text_transform: Optional[Callable] = None,
        image_transforms: Optional[Tuple[Callable, Callable]] = None,
        mlm_probablity: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        finetuning: bool = False,
        ignore_index: int = -1,
        itm_probability: float = 0.1,
        allow_uneven_batches: bool = False,
        fetch_num_threads: int = 4,
        fetch_retries: int = 0,
        fetch_sleep_timer: int = 0,
        fetch_timeout: Optional[float] = None,
        fetch_batch_size: int = 50,
        **kwargs,
    ):
        super().__init__()

        self.train_dataset_infos = train_infos
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos
        if image_transforms is None:
            if not finetuning:
                image_transforms = default_image_pretraining_transforms()
            else:
                image_transforms = default_torchvision_transforms(use_dict=True)

        self.train_image_transform, self.test_image_transform = image_transforms
        self.text_transform = text_transform
        self.mlm_probability = mlm_probablity
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_index = ignore_index
        self.itm_probability = itm_probability
        self.allow_uneven_batches = allow_uneven_batches
        self.fetch_num_threads = fetch_num_threads
        self.fetch_retries = fetch_retries
        self.fetch_sleep_timer = fetch_sleep_timer
        self.fetch_timeout = fetch_timeout
        self.fetch_batch_size = fetch_batch_size

    def setup(self, stage=None):
        if self.text_transform is None:
            # TODO Update to use whole word mask vocab
            text_tokenizer = BertTokenizer.from_pretrained(
                TEXT_WHOLE_WORD_MASK_TOKENIZER
            )
            self.text_transform = default_text_transform(
                text_tokenizer, max_text_length=VL_MAX_LENGTH_DEFAULT
            )
        self.text_tokenizer = self.text_transform.keywords["tokenizer"]
        train_vl_transform = VLTransform(
            self.train_image_transform, self.text_transform, unnest=True
        )
        val_vl_transform = VLTransform(self.test_image_transform, self.text_transform, unnest=True)

        self.train_dataset = ISICDataset(self.train_dataset_infos[0]["csv_path"],
                                         self.train_dataset_infos[0]["img_path"],
                                         self.train_dataset_infos[0]["mode"])
        self.train_dataset.set_transform(
            partial(
                train_vl_transform,
                dataset=self.train_dataset,
                itm_probability=self.itm_probability,
            )
        )

        self.val_dataset = ISICDataset(self.val_dataset_infos[0]["csv_path"],
                                       self.val_dataset_infos[0]["img_path"],
                                       self.val_dataset_infos[0]["mode"])
        self.val_dataset.set_transform(
            partial(
                val_vl_transform,
                dataset=self.val_dataset,  # Pass a copy to transform
                itm_probability=self.itm_probability,
            )
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=True,
            collate_fn=self._build_collator(),
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=False,
            collate_fn=self._build_collator(),
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            drop_last=True,
        )

    def _build_collator(self):
        return DataCollatorForWholeWordMaskRetainingBatch(
            self.text_tokenizer, mlm_probability=self.mlm_probability
        )

    def on_before_batch_transfer(self, batch, *args):
        batch.pop("token_type_ids", None)
        mask = batch.pop("attention_mask", None)
        if (
            mask is not None
            and mask.size(0) < self.batch_size
            and not self.allow_uneven_batches
        ):
            batch = pad_batch(batch, self.batch_size)
        return batch

    def on_after_batch_transfer(self, batch, *args):
        text_masked = batch.pop("input_ids")
        mlm_labels = batch.pop("labels", None)
        mlm_labels[mlm_labels == -100] = self.ignore_index
        text = text_masked.detach().clone()
        text[mlm_labels != -1] = mlm_labels[mlm_labels != -1]
        batch.update(
            {"mlm_labels": mlm_labels, "text": text, "text_masked": text_masked}
        )
        return batch



class ISICDataModule(LightningDataModule):
    def __init__(
        self,
        train_infos: List[ISICInfo],
        val_infos: Optional[List[ISICInfo]] = None,
        transforms: Optional[Tuple[Callable, Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        allow_uneven_batches: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.train_dataset_infos = train_infos
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_uneven_batches = allow_uneven_batches

        if transforms is None:
            transforms = default_image_pretraining_transforms()

        self.train_transform, self.test_transform = transforms

    def setup(self, stage=None):
        train_transform = self.train_transform
        val_transform = self.test_transform

        self.train_dataset = ISICDataset(self.train_dataset_infos[0]["csv_path"],
                                         self.train_dataset_infos[0]["img_path"],
                                         self.train_dataset_infos[0]["mode"])

        self.train_dataset.set_transform(train_transform)
        self.val_dataset = ISICDataset(self.val_dataset_infos[0]["csv_path"],
                                       self.val_dataset_infos[0]["img_path"],
                                       self.val_dataset_infos[0]["mode"])
        self.val_dataset.set_transform(val_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=True,
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            # ideally, we don't need to drop these for unimodal cases
            # but just to be safe
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=False,
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            # ideally, we don't need to drop these for unimodal cases
            # but just to be safe
            drop_last=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def on_before_batch_transfer(self, batch, *args):
        if batch["label"].size(0) < self.batch_size and not self.allow_uneven_batches:
            batch = pad_batch(batch, self.batch_size)
        return batch



class TorchVisionDataModule(LightningDataModule):
    def __init__(
        self,
        train_infos: List[TorchVisionDatasetInfo],
        # Val info is not used for torchvision datamodule, but kept to keep things consistent
        val_infos: Optional[List[TorchVisionDatasetInfo]] = None,
        dataset_root: Optional[str] = None,
        image_transforms: Optional[Tuple[Callable, Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs: Any,
    ):
        super().__init__()
        self.train_info = train_infos[0]
        if val_infos is None:
            val_infos = train_infos
        self.val_info = val_infos[0]

        self.train_class_ptr, self.train_root = self._parse_info(
            self.train_info, dataset_root=dataset_root
        )
        self.val_class_ptr, self.val_root = self._parse_info(
            self.val_info, dataset_root=dataset_root
        )

        if image_transforms is None:
            image_transforms = default_torchvision_transforms()

        self.train_transform, self.test_transform = image_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _parse_info(
        self, info: TorchVisionDatasetInfo, dataset_root: Optional[str] = None
    ):
        assert hasattr(
            torchvision.datasets, info.key
        ), f"No dataset named {info.key} present in torchvision.datasets"
        class_ptr = getattr(torchvision.datasets, info.key)
        if dataset_root is None:
            dataset_root = os.path.join(TRANSFORMERS_CACHE, "datasets", "torchvision")
            dataset_root = os.path.join(dataset_root, class_ptr.__name__.lower())
            os.makedirs(dataset_root, exist_ok=True)

        return class_ptr, dataset_root

    def setup(self, stage=None):
        self.train_dataset = self.train_class_ptr(
            self.train_root,
            split=self.train_info.train_split,
            transform=self.train_transform,
            download=True,
        )

        if self.val_info.has_val:
            self.val_dataset = self.val_class_ptr(
                self.val_root,
                split=self.val_info.val_split,
                transform=self.test_transform,
                download=True,
            )

        self.test_dataset = self.val_class_ptr(
            self.val_root,
            split=self.val_info.test_split,
            transform=self.test_transform,
            download=True,
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset)

    def val_dataloader(self):
        if self.val_info.has_val:
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        return self._build_dataloader(dataset, shuffle=False)

    def test_dataloader(self):
        return self._build_dataloader(self.test_dataset, shuffle=False)

    def _build_dataloader(self, dataset: torch.utils.data.Dataset, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def on_before_batch_transfer(self, batch, *args):
        images, targets = batch
        batch = {"image": images, "labels": targets}
        return batch
