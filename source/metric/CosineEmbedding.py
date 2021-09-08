import torch
from torchmetrics import Metric


class CosineEmbedding(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(CosineEmbedding, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.cossim = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, r1, r2, cls):
        self.correct += 1-self.cossim(r1,r2,cls)
        self.total += 1

    def compute(self):
        return self.correct / self.total