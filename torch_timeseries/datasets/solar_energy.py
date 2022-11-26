import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import torch
from typing import Callable, List, Optional
import os
import resource
import numpy as np
from.dataset import Dataset, TimeSeriesDataset


class SolarEnergyV2(TimeSeriesDataset):
    name: str = 'solar_AL'
    num_features: int = 137
    sample_rate: int  # in munites

    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz",
            self.raw_dir,
            filename="solar_AL.txt.gz",
            md5="41ef7fdc958c2ca3fac9cd06d6227073",
        )

    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.raw_dir, 'solar_AL.txt')
        self.raw_data = np.loadtxt(self.file_name, delimiter=',')
        return self.raw_data

    def _process(self) -> np.ndarray:
        return super()._process()


class SolarEnergy(Dataset):

    tasks = ['supervised', 'prediction', 'multi_timeseries', 'regression']

    url = "https://github.com/laiguokun/multivariate-time-series-data"

    feature_nums = 137
    resources = {
        'solar_AL.txt.gz': '41ef7fdc958c2ca3fac9cd06d6227073'
    }

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        """
        data from this github repo: https://github.com/laiguokun/multivariate-time-series-data

        Args:
            root (str): the data directory to save
            transform (Optional[Callable], optional): . Defaults to None.
            pre_transform (Optional[Callable], optional): . Defaults to None.
        """
        super().__init__(root, transform, pre_transform)

        self.dataset_name = 'solar_AL'

        self.raw_dir = os.path.join(root, self.dataset_name, 'raw',)
        self.processed_dir = os.path.join(root, self.dataset_name, 'processed')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.download()

    def raw_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.raw_dir, 'solar_AL.txt'), sep=',', header=None)

    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz",
            self.raw_dir,
            filename="solar_AL.txt.gz",
            md5="41ef7fdc958c2ca3fac9cd06d6227073",
        )
