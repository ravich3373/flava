# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from functools import partial
from typing import Any, Callable, Optional

import torch
from torchmultimodal.transforms.flava_transform import FLAVAImageTransform
from torchvision import transforms
from transformers import BertTokenizer


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGE_DEFAULT_SIZE = (224, 224)
VL_MAX_LENGTH_DEFAULT = 77
TEXT_MAX_LENGTH_DEFAULT = 512
TEXT_DEFAULT_TOKENIZER = "bert-base-uncased"
TEXT_WHOLE_WORD_MASK_TOKENIZER = "bert-large-uncased-whole-word-masking"


def encode_text(text, tokenizer, *args, **kwargs):
    return tokenizer(text, *args, **kwargs)


def encode_text_batch(
    batch, tokenizer, text_columns, return_batch=False, *args, **kwargs
):
    texts = [batch[column] for column in text_columns]
    tokens = tokenizer(*texts, *args, **kwargs)
    if return_batch:
        batch.update(tokens)
        return batch
    return tokens


def transform_image_dict(image_dict, transform, *args, **kwargs):
    return {"image": transform(image_dict["image"], *args, **kwargs)}

def transform_image_dict_ravi(image, transform, *args, **kwargs):
    return {"image": transform(image, *args, **kwargs)}

def default_torchvision_transforms(
    size=IMAGE_DEFAULT_SIZE,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    use_dict=False,
    inp_dict=True
):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    if use_dict:
        if inp_dict:
            transform = partial(transform_image_dict, transform=transform)
        else:
            transform = partial(transform_image_dict_ravi, transform=transform)

    return transform, transform


def default_image_pretraining_transforms():
    return FLAVAImageTransform(encoder_input_size=384,mask_window_size=24, codebook_input_size=192), FLAVAImageTransform(encoder_input_size=384,mask_window_size=24, codebook_input_size=192, is_train=False)


def default_text_transform(
    text_tokenizer: Optional[Callable] = None,
    max_text_length: int = TEXT_MAX_LENGTH_DEFAULT,
    **kwargs: Any,
):
    if text_tokenizer is None:
        text_tokenizer = BertTokenizer.from_pretrained(TEXT_DEFAULT_TOKENIZER)

    text_transform = partial(
        encode_text,
        tokenizer=text_tokenizer,
        padding="max_length",
        max_length=max_text_length,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    return text_transform


def default_vl_text_transform(
    text_tokenizer: Optional[Callable] = None,
    max_text_length: int = VL_MAX_LENGTH_DEFAULT,
    **kwargs: Any,
):
    if text_tokenizer is None:
        text_tokenizer = BertTokenizer.from_pretrained(TEXT_WHOLE_WORD_MASK_TOKENIZER)
    return default_text_transform(text_tokenizer, max_text_length=max_text_length)


def pad_batch(batch, batch_size):
    for item in batch.keys():
        if isinstance(batch[item], torch.Tensor):
            diff = batch_size - batch[item].size(0)
            pad = batch[item][-diff:].detach().clone()
            batch[item] = torch.cat([batch[item], pad], dim=0)
    return batch


class VLTransform:
    def __init__(self, image_transform, text_transform, unnest=False, use_dict=False):
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.unnest = unnest
        self.use_dict = use_dict

    def __call__(self, info, dataset, itm_probability):
        output = {}
        text = info["text"]
        image = info["image"]
        if itm_probability > 0:
            output["itm_labels"] = torch.ones((1), dtype=torch.long)

        if random.random() < itm_probability:
            while text == info["text"]:
                text = dataset.select([random.randint(0, len(dataset) - 1)])[0]["text"]
            output["itm_labels"] = torch.zeros((1), dtype=torch.long)

        if self.unnest: # ravi needed to work with unnest
            text = [text]
            if not self.use_dict:
                image = [image]
        if self.use_dict:
            output.update({"image": self.image_transform(image)})
        else:
            output.update(self.image_transform(image))
        output["labels"] = torch.Tensor([info["label"]]).long()
        output.update(self.text_transform(text))
        # ravi, vl to get mm embedding check for text key
        output["text"] = output["input_ids"]
        #
        if self.unnest:
            return self.unnester(output)
        return output

    def unnester(self, py_dict):
        if not self.use_dict:
            return {key: array[0] for key, array in py_dict.items()}
        else:
            ret = {}
            for key, array in py_dict.items():
                if key == "image":
                    ret[key] = array
                else:
                    ret[key] = array[0]
            return ret
