from typing import Any, Callable, Optional
from torch import Tensor
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    feature_nums = 0    
    def __init__(self,root:str,transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,single_step=True):
        """_summary_

        Args:
            root (str): data save location
            transform (Optional[Callable], optional): _description_. Defaults to None.
            target_transform (Optional[Callable], optional): _description_. Defaults to None.
            single_step (bool, optional): True for single_step data, False for multi_steps data. Defaults to True.
        """
        super().__init__()
        
        
        
    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError
    
    
    
    def raw_df(self):
        raise NotImplementedError()
    

    
    def __getitem__(self, index: Any):
        
        return super().__getitem__()
    
    
    