import torch
from pytorch_lightning.metrics import Metric


class CosineEmbedding(Metric):
    def __init__(self):
        super().__init__()
        self.cossim = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
        self.add_state("cossims", default=[])


    def update(self, r1, r2, cls):

        self.cossims.append(1-self.cossim(r1,r2,cls))

    def compute(self):
        return torch.mean(torch.tensor(self.cossims))