import torch

from source.model.SiameseModel import SiameseModel
import os

import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity




def get_tokenizer(params):
    tokenizer = AutoTokenizer.from_pretrained(
        params.tokenizer.architecture
    )

    return tokenizer


@hydra.main(config_path="settings/", config_name="settings.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)
    z_shot_sentence_similarity(params)


def encode (text, tokenizer, max_length):
    return tokenizer.encode(text=text, max_length=max_length, padding="max_length",
                                 truncation=True, return_tensors="pt")

def rpr(text, model):
    return torch.squeeze(
        model(text)).tolist()


def z_shot_sentence_similarity(params):
    tokenizer = get_tokenizer(params.model)
    model = SiameseModel(params.model)

    sentences = ['The cat sits outside',
                 'I love pasta',
                 'The cat plays in the garden',
                 'Do you like pizza?']

    senteces_tkd = [
            encode(sentece, tokenizer, 16) for sentece in sentences
    ]

    senteces_rep = [
        rpr(sentece, model) for sentece in senteces_tkd
    ]

    sims = cosine_similarity(
        senteces_rep,
        senteces_rep
    )



    return sims


if __name__ == '__main__':
    perform_tasks()