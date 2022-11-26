import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import torch
from typing import Any, Callable, List, Optional
import os
import resource
from.dataset import Dataset, TimeSeriesDataset


class ExchangeRateV2(TimeSeriesDataset):
    name: str = 'exchange_rate'
    num_features: int = 8
    sample_rate: int  # in munites

    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
            self.raw_dir,
            filename="exchange_rate.txt.gz",
            md5="9dd5a9c8f8f324e234938400f232fa08",
        )

    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.raw_dir, 'exchange_rate.txt')
        self.raw_data = np.loadtxt(self.file_name, delimiter=',')
        return self.raw_data

    def _process(self) -> np.ndarray:
        return super()._process()


class ExchangeRate(Dataset):

    tasks = ['supervised', 'prediction', 'multi_timeseries', 'regression']

    url = "https://github.com/laiguokun/multivariate-time-series-data"
    feature_nums = 8
    resources = {
        'exchange_rate.txt.gz': '9dd5a9c8f8f324e234938400f232fa08'
    }

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, window=10, horizon=1):
        """
        data from this github repo: https://github.com/laiguokun/multivariate-time-series-data

        Args:
            root (str): the data directory to save
            transform (Optional[Callable], optional): . Defaults to None.
            pre_transform (Optional[Callable], optional): . Defaults to None.
        """
        super().__init__(root, transform, pre_transform)

        self.dataset_name = 'exchange_rate'

        self.raw_dir = os.path.join(root, self.dataset_name, 'raw',)
        self.processed_dir = os.path.join(root, self.dataset_name, 'processed')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.window = window
        self.horizon = horizon

        self.file_name = 'exchange_rate.txt'
        self.raw_data = np.loadtxt(self.file_name, delimiter=',')
        self.raw_tensor = torch.from_numpy(self.raw_data)
        self.tensor = torch.from_numpy(self.raw_data)

        self.num_nodes = self.feature_nums

        self.download()

    def __len__(self):
        return len(self.tensor) - self.window - self.horizon + 1

    def __getitem__(self, index: Any):
        return self.tensor[index:index+self.window], self.tensor[self.window + self.horizon - 1 + index]

    def raw_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.raw_dir, 'exchange_rate.txt'), sep=',', header=None)

    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
            self.raw_dir,
            filename="exchange_rate.txt.gz",
            md5="9dd5a9c8f8f324e234938400f232fa08",
        )
