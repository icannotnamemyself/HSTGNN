from typing import Callable, Optional
from torch import Tensor
import torch.utils.data


class PytDataset(torch.utils.data.Dataset):
    
    def __init__(self,root:str,transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None):
        super().__init__()