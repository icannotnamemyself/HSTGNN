import os
import resource
from.PytDataset import PytDataset
from torch_timeseries.data.extract import extract_zip
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class ExchangeRate(PytDataset):

    tasks =['supervised', 'prediction', 'multi_timeseries', 'regression']
    
    url = "https://github.com/laiguokun/multivariate-time-series-data"

    resources = {
        'exchange_rate.txt.gz': '9dd5a9c8f8f324e234938400f232fa08'
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

        self.dataset_name = 'exchange_rate'

        self.raw_dir = os.path.join(root, self.dataset_name, 'raw',)
        self.processed_dir = os.path.join(root, self.dataset_name, 'processed')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.download()

    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
            self.raw_dir,
            filename="exchange_rate.txt.gz",
            md5="9dd5a9c8f8f324e234938400f232fa08",
        )
        
        
