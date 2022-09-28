from typing import Callable, Optional
from torch import Tensor
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self,root:str,transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None):
        super().__init__()
        
        
    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError
    
    
    
    def raw_df(self):
        raise NotImplementedError()