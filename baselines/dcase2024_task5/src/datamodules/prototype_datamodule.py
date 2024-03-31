from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class PrototypeDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_param: dict = {},
        eval_param: dict = {},
        path: dict = {},
        features: dict = {},
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.init()

    def setup(self, stage: Optional[str] = None):
        pass

    def init(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        from src.datamodules.components import (
            IdentityBatchSampler,
            PrototypeDynamicDataSet,
            PrototypeDynamicArrayDataSet,  # The PCEN training dataset
            PrototypeDynamicArrayDataSetVal,  # The validation dataset for validation process
            PrototypeDynamicArrayDataSetWithEval,  # Training with the first 5 validation data
        )

        # Training dataset
        if self.hparams.train_param.use_validation_first_5:
            self.dataset = PrototypeDynamicArrayDataSetWithEval(
                path=self.hparams.path,
                features=self.hparams.features,
                train_param=self.hparams.train_param,
            )
        else:
            self.dataset = PrototypeDynamicArrayDataSet(
                path=self.hparams.path,
                features=self.hparams.features,
                train_param=self.hparams.train_param,
            )

        self.sampler = IdentityBatchSampler(
            self.hparams.train_param,
            self.dataset.train_eval_class_idxs,
            self.dataset.extra_train_class_idxs,
            batch_size=self.hparams.train_param.n_shot * 2,
            n_episode=int(
                len(self.dataset)
                / (self.hparams.train_param.k_way * self.hparams.train_param.n_shot * 2)
            ),
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=self.sampler, num_workers=2
        )

        # Validation dataset
        self.val_dataset = PrototypeDynamicArrayDataSetVal(
            path=self.hparams.path,
            features=self.hparams.features,
            train_param=self.hparams.train_param,
        )
        self.val_sampler = IdentityBatchSampler(
            self.hparams.train_param,
            self.val_dataset.eval_class_idxs,
            [],
            batch_size=self.hparams.train_param.n_shot * 2,
            n_episode=int(
                len(self.val_dataset)
                / (self.hparams.train_param.k_way * self.hparams.train_param.n_shot * 2)
            ),
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_sampler=self.val_sampler, num_workers=2
        )

        from src.datamodules.components import (
            PrototypeTestSet,
            PrototypeAdaSeglenTestSet,
            PrototypeAdaSeglenBetterNegTestSetV2,
        )

        if self.hparams.train_param.adaptive_seg_len:
            self.data_test = PrototypeAdaSeglenBetterNegTestSetV2(
                self.hparams.path,
                self.hparams.features,
                self.hparams.train_param,
                self.hparams.eval_param,
            )
        else:
            self.data_test = PrototypeTestSet(
                self.hparams.path,
                self.hparams.features,
                self.hparams.train_param,
                self.hparams.eval_param,
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )
