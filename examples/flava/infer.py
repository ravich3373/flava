import torch
from torchmultimodal.models.flava.model import flava_model_for_classification, flava_model_for_pretraining
from torchvision.transforms import ToPILImage
from transformers import BertTokenizer
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from flava.model import FLAVAPreTrainingLightningModule
from flava.data.datamodules import ISICDataset
from flava.data.transforms import default_image_pretraining_transforms


def main():
    tokernizer = BertTokenizer.from_pretrained("bert-base-uncased")
    a = torch.load("flava-epoch=00-step=21000.ckpt", map_location=torch.device('cpu'))

    m = FLAVAPreTrainingLightningModule()
    m.load_state_dict(a['state_dict'])

    ds = ISICDataset("/home/ravi/Documents/courses/healthcare/Research/test_data.csv",
                 "/home/ravi/Documents/courses/healthcare/Research/archive/train",
                 "image",
                  True)
    tfms = default_image_pretraining_transforms()[1]
    ds.set_transform(tfms)
    dl = torch.utils.data.DataLoader(
                ds,
                batch_size=1,
                num_workers=1,
                sampler=None,
                shuffle=False,
                # collate_fn=_build_collator(),
                # uneven batches can cause distributed issues,
                # drop last batch to prevent those.
                drop_last=False,
            )
    to_pil = ToPILImage()

    embeddings = []
    labels = []
    file_paths = []
    for data in dl:
        r = m._step(data, 1)
        embeds = m.model.model.image_encoder(data['image']).last_hidden_state[:, 0].detach().cpu().squeeze().numpy()
        y = data['label'].detach().cpu().numpy()
        im_pths = data['img_pth']

        embeddings.append(embeds)
        labels.append(y)
        file_paths.extend(im_pths)
    
    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels)

    embed_2d = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=3).fit_transform(embeddings)
    
    dict = {"embeddings": embeddings,
            "labels": labels,
            "image": file_paths,
            "embed_2d": embed_2d}
    
    df = pd.DataFrame.from_dict(dict)
    df.to_csv("infer_tsne.csv")
