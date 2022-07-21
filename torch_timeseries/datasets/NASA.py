import os
import resource
from torch_timeseries.data import Dataset, download_url
from torch_timeseries.data.extract import extract_zip
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity



class NASA(Dataset):
    
    url = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
    
    
    resources = {
        'data.zip': 'c40a236775c1c3a26de601f66541b414'
    }
    
    def __init__(self, root: str, name: str, split: str = "full",
                num_train_per_class: int = 20, num_val: int = 500,
                num_test: int = 1000, transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None):
        """NASA 

        Args:
            root (str): _description_
            name (str): _description_
            split (str, optional): _description_. Defaults to "public".
            num_train_per_class (int, optional): _description_. Defaults to 20.
            num_val (int, optional): _description_. Defaults to 500.
            num_test (int, optional): _description_. Defaults to 1000.
            transform (Optional[Callable], optional): _description_. Defaults to None.
            pre_transform (Optional[Callable], optional): _description_. Defaults to None.
        """
        super().__init__(root, transform, pre_transform)
        
        
        self.name = name
        self.dataset_name = 'NASA'
        self.split = split.lower()
        assert self.split in ['full']
        
        
        self.raw_dir = os.path.join(root,self.dataset_name, 'raw',)
        self.processed_dir = os.path.join(root,self.dataset_name, 'processed')
        
        
        # 文件存在且完整性通过则跳过下载
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.download()
    
    def download(self) -> None:
        download_and_extract_archive(
            "https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
            self.raw_dir,
            filename="data.zip",
            md5="c40a236775c1c3a26de601f66541b414",
        )
        


