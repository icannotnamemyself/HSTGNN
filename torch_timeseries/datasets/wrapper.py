from typing import TypeVar

import pandas as pd
from torch_timeseries.data.scaler import MaxAbsScaler, Scaler

from torch_timeseries.utils.timefeatures import time_features
from .dataset import TimeSeriesDataset
from torch.utils.data import Dataset
import numpy as np
import torch


class MultiStepTimeFeatureSet(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, scaler: Scaler, time_enc=0, window: int = 168, horizon: int = 3, steps: int = 2, freq=None, scaler_fit=True):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps
        self.time_enc = time_enc
        self.scaler = scaler
        if scaler_fit:
            self.scaler.fit(self.dataset.data)
        self.scaled_data = self.scaler.transform(self.dataset.data)

        self.num_features = self.dataset.num_features
        if freq is None:
            self.freq = self.dataset.freq
        else:
            self.freq = freq
        self.length = self.dataset.length

    def transform(self, values):
        return self.scaler.transform(values)

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)

    def __getitem__(self, index):
        # 如果是数字索引，则返回单个数据项
        if isinstance(index, int):
            x = self.scaled_data[index:index+self.window]
            x_date_enc = time_features(
                self.dataset.df.iloc[index:index+self.window],
                self.time_enc, self.freq)
            y = self.scaled_data[self.window + self.horizon - 1 +
                                 index:self.window + self.horizon - 1 + index+self.steps]
            y_date_enc = time_features(
                self.dataset.df.iloc[self.window + self.horizon -
                                     1 + index:self.window + self.horizon - 1 + index+self.steps], self.time_enc, self.freq)
            return x, y, x_date_enc, y_date_enc
        else:
            raise TypeError('Not surpported index type!!!')

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1


class SingleStepWrapper(Dataset):

    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        return self.dataset.data[index:index+self.window], self.dataset.data[self.window + self.horizon - 1 + index]

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1


class MultiStepWrapper(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3, steps: int = 2):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps

    def __getitem__(self, index):
        return self.dataset.data[index:index+self.window], self.dataset.data[self.window + self.horizon - 1 + index:self.window + self.horizon - 1 + index+self.steps]

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1


class SingStepFlattenWrapper(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.dataset.data[index:index+self.window]
        y = self.dataset.data[self.window + self.horizon - 1 + index]
        return x.flatten(), y

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1


class MultiStepFlattenWrapper(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3, steps: int = 2):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps

    def __getitem__(self, index):
        x = self.dataset.data[index:index+self.window]
        y = self.dataset.data[self.window + self.horizon - 1 +
                              index:self.window + self.horizon - 1 + index+self.steps]
        return x.flatten(), y.flatten()

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1
