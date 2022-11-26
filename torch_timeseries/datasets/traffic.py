import os
import resource
from .dataset import Dataset, TimeSeriesDataset
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np

class TrafficV2(TimeSeriesDataset):
    name:str= 'traffic'
    num_features: int = 862
    sample_rate:int # in munites
    
    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
            self.raw_dir,
            filename="traffic.txt.gz",
            md5="db745d0c9f074159581a076cbb3f23d6",
        )
        
    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.raw_dir, 'traffic.txt')
        self.raw_data = np.loadtxt(self.file_name, delimiter=',')
        return self.raw_data
        
    def _process(self) -> np.ndarray:
        return super()._process()

class Traffic(Dataset):

    tasks =['supervised', 'prediction', 'multi_timeseries', 'regression']
    
    url = "https://github.com/laiguokun/multivariate-time-series-data"
    feature_nums = 862
    resources = {
        'traffic.txt.gz': 'db745d0c9f074159581a076cbb3f23d6'
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

        self.dataset_name = 'traffic'

        self.raw_dir = os.path.join(root, self.dataset_name, 'raw',)
        self.processed_dir = os.path.join(root, self.dataset_name, 'processed')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.download()

    
    def raw_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.raw_dir, 'traffic.txt'), sep=',', header=None)
    
    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
            self.raw_dir,
            filename="traffic.txt.gz",
            md5="db745d0c9f074159581a076cbb3f23d6",
        )
        
        
