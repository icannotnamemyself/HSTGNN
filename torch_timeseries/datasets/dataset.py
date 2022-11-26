from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Any, Callable, Generic, NewType, Optional, TypeVar, Union
from torch import Tensor
import torch.utils.data
import os
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from abc import ABC, abstractmethod


class Dataset(torch.utils.data.Dataset):
    feature_nums = 0

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, single_step=True):
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



@dataclass
class TimeSeriesDatasetDescription:
    name: str
    num_features: int
    sample_rate: int


# StoreTypes = Union[np.ndarray, Tensor]
StoreTypes = np.ndarray


class TimeSeriesDataset(ABC):
    name: str
    num_features: int
    sample_rate: int

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        """_summary_

        Args:
            root (str): data save location
            transform (Optional[Callable], optional): . Defaults to None.
            pre_transform (Optional[Callable], optional): . Defaults to None.

        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform

        self.raw_dir = os.path.join(root, self.name, 'raw',)
        self.processed_dir = os.path.join(root, self.name, 'processed')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.download()
        self.raw_data = self._load()  # load to self.data
        self.data = self._process()  # process data may save to processed_dir

    @abstractmethod
    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    @abstractmethod
    def _load(self) -> StoreTypes:
        """Loads the dataset to the :attr:`self.raw_data` .

        Raises:
            NotImplementedError: _description_

        Returns:
            T: should return a numpy.array or torch.tensor or pandas.Dataframe
        """

        raise NotImplementedError

    @abstractmethod
    def _process(self) -> StoreTypes:
        r"""Downloads the dataset to the :attr:`self.data` ."""
        data = self.raw_data
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        if self.transform is not None:
            data = self.transform(data)
        return data
