
from abc import abstractclassmethod
import os.path as osp
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import numpy as np
import torch.utils.data
from torch import Tensor

# from torch_geometric.data import Data
from torch_timeseries.data.makedirs import makedirs

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class Dataset(torch.utils.data.Dataset):
    
    @property
    def raw_file_dict(self) -> Dict[str, Dict]:
        r"""The file tree of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        raise NotImplementedError

    @property
    def processed_file_dict(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        raise NotImplementedError
    
    @abstractclassmethod
    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError
    
    @abstractclassmethod
    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError
    
    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__()
        
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))
        
        self.root = root
        self.dataset_name = 'default'
        self.raw_dir = osp.join(root,self.dataset_name, 'raw')
        self.processed_dir = osp.join(root,self.dataset_name, 'processed')
        
        if osp.exists(self.raw_dir):  # pragma: no cover
            return
        makedirs(self.raw_dir)
        self.download()

        self.process()


