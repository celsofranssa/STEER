import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.dataset.RQEDataset import RQEDataset


class RQEDataModule(pl.LightningDataModule):
    """

    """

    def __init__(self, params, tokenizer, fold):
        super(RQEDataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.fold = fold

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = RQEDataset(
                path=f"{self.params.dir}fold_{self.fold}/train.jsonl",
                tokenizer=self.tokenizer,
                max_length=self.params.max_length
            )

            self.val_dataset = RQEDataset(
                path=f"{self.params.dir}fold_{self.fold}/test.jsonl",
                tokenizer=self.tokenizer,
                max_length=self.params.max_length
            )

        if stage == 'test' or stage is None:
            self.test_dataset = RQEDataset(
                path=f"{self.params.dir}fold_{self.fold}/test.jsonl",
                tokenizer=self.tokenizer,
                max_length=self.params.max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )
