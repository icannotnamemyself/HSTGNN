from typing import Sequence, Tuple, Type

import torch
from torch_timeseries.data.scaler import Scaler
from torch_timeseries.datasets.dataset import (
    Dataset,
    TimeSeriesDataset,
    TimeseriesSubset,
)
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet


class ChunkSequenceTimefeatureDataLoader:
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        time_enc=0,
        window: int = 168,
        horizon: int = 3,
        steps: int = 2,
        scale_in_train=False,
        shuffle_train=True,
        freq=None,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        num_worker: int = 3,
    ) -> None:
        """

        Split the dataset sequentially, and then randomly sample from each subset.

        :param dataset: the input dataset, must be of type datasets.Dataset
        :param train_ratio: the ratio of the training set
        :param test_ratio: the ratio of the testing set
        :param val_ratio: the ratio of the validation set
        :param independent_scaler: whether to set independent scaler for train , val and test dataset,
                default: False, will have a global scaler for all data
                if set to True, scaler is fitted by differenct part of data
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - self.train_ratio - self.val_ratio

        assert (
            self.train_ratio + self.val_ratio + self.test_ratio == 1.0
        ), "Split ratio must sum up to 1.0"
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataset = dataset

        self.scaler = scaler
        self.window = window
        self.freq = freq
        self.time_enc = time_enc
        self.steps = steps
        self.horizon = horizon
        self.shuffle_train = shuffle_train
        self.scale_in_train = scale_in_train

        self._load()

    def _load(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return the splitted training, testing and validation dataloders

        :return: a tuple of train_dataloader, test_dataloader and val_dataloader
        """
        # fixed suquence dataset
        indices = range(0, len(self.dataset))

        train_size = int(self.train_ratio * len(self.dataset))
        test_size = int(self.val_ratio * len(self.dataset))
        val_size = len(self.dataset) - test_size - train_size
        train_subset = TimeseriesSubset(self.dataset, indices[0:train_size])
        val_subset = TimeseriesSubset(
            self.dataset, indices[train_size : (test_size + train_size)]
        )

        test_subset = TimeseriesSubset(self.dataset, indices[-val_size:])
        if self.scale_in_train:
            self.scaler.fit(train_subset)
        else:
            self.scaler.fit(self.dataset.data)

        train_dataset = MultiStepTimeFeatureSet(
            train_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
            steps=self.steps,
            freq=self.freq,
            scaler_fit=False,
        )
        val_dataset = MultiStepTimeFeatureSet(
            val_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
            steps=self.steps,
            freq=self.freq,
            scaler_fit=False,
        )
        test_dataset = MultiStepTimeFeatureSet(
            test_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
            steps=self.steps,
            freq=self.freq,
            scaler_fit=False,
        )

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(test_dataset)
        # RandomSampler 与 Dataloader generator都需要设置，否则还是无法复现
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_worker,
            
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        return train_loader, test_loader, val_loader