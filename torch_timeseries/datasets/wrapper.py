from typing import TypeVar
from .dataset import TimeSeriesDataset
from torch.utils.data import Dataset as DatasetWrapper
import numpy as np
import torch
T_co = TypeVar('T_co', covariant=True)


class SingleStepWrapper(DatasetWrapper):

    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        return self.dataset.data[index:index+self.window], self.dataset.data[self.window + self.horizon - 1 + index]

    def __len__(self):
        return len(self.dataset.data) - self.window - self.horizon + 1


class MultiStepWrapper(DatasetWrapper):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3, steps: int = 2):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps

    def __getitem__(self, index):
        return self.dataset.data[index:index+self.window], self.dataset.data[self.window + self.horizon - 1 + index:self.window + self.horizon - 1 + index+self.steps]

    def __len__(self):
        return len(self.dataset.data) - self.window - self.horizon + 1 - self.steps + 1


class SingStepFlattenWrapper(DatasetWrapper):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.dataset.data[index:index+self.window]
        y = self.dataset.data[self.window + self.horizon - 1 + index]
        return x.flatten(), y

    def __len__(self):
        return len(self.dataset.data) - self.window - self.horizon + 1


class MultiStepFlattenWrapper(DatasetWrapper):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3, steps: int = 2):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps

    def __getitem__(self, index):
        x = self.dataset.data[index:index+self.window]
        y = self.dataset.data[self.window + self.horizon - 1 +
                              index:self.window + self.horizon - 1 + index+self.steps]
        return x.flatten(), y

    def __len__(self):
        return len(self.dataset.data) - self.window - self.horizon + 1 - self.steps + 1
