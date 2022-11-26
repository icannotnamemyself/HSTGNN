import os
import resource
from.dataset import Dataset, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data



class ElectricityV2(TimeSeriesDataset):
    name:str= 'electricity'
    num_features: int = 321
    sample_rate:int # in munites
    
    def download(self):
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
            self.raw_dir,
            filename="electricity.txt.gz",
            md5="07d51dc39c404599ead932937985957b",
        )
        
    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.raw_dir, 'electricity.txt')
        self.raw_data = np.loadtxt(self.file_name, delimiter=',')
        return self.raw_data
    
    def _process(self) -> np.ndarray:
        return super()._process()
    
class Electricity(Dataset):

    tasks =['supervised', 'prediction', 'multi_timeseries', 'regression']
    
    url = "https://github.com/laiguokun/multivariate-time-series-data"
    feature_nums = 321
    resources = {
        'electricity.txt.gz': '07d51dc39c404599ead932937985957b'
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
        self.window = window
        self.horizon = horizon
        # TODO: 解决感受野 大于 时间长度时的问题
        
        self.dataset_name = 'electricity'


        
        self.raw_dir = os.path.join(root, self.dataset_name, 'raw',)
        self.processed_dir = os.path.join(root, self.dataset_name, 'processed')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.download()
        
        self.file_name = os.path.join(self.raw_dir, 'electricity.txt')
        self.raw_data = np.loadtxt(self.file_name, delimiter=',')
        self.raw_tensor = torch.from_numpy(self.raw_data)
        self.tensor = torch.from_numpy(self.raw_data)
        
        self.num_nodes = len(self.raw_df().columns)

        
    # def __process(self):
    #     for i in range(0, len(self.raw_tensor) -self.window - self.horizon + 1):
    #         self.X[i] = self.raw_tensor[i:i+self.window]
            # self.Y[i] = self.raw_tensor[self.window + self.horizon - 1 + i]
        
    def __len__(self):
        return len(self.tensor) -self.window - self.horizon + 1
        
    def __getitem__(self, index: Any):
        return self.tensor[index:index+self.window] , self.tensor[self.window + self.horizon - 1 + index]
        
    def raw_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.raw_dir, 'electricity.txt'), sep=',', header=None)
    
    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
            self.raw_dir,
            filename="electricity.txt.gz",
            md5="07d51dc39c404599ead932937985957b",
        )
        
        
    
        