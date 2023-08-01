

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Any, Callable, Generic, NewType, Optional, Sequence, TypeVar, Union
from torch import Tensor
import torch.utils.data
import os
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from abc import ABC, abstractmethod

from torch_timeseries.data.scaler import MaxAbsScaler, Scaler, StoreType


from enum import Enum, unique

from torch_timeseries.datasets.dataset import Freq, TimeSeriesDataset

class Dummy(TimeSeriesDataset):
    name: str = 'dummy'
    num_features:int = 2
    freq : Freq = Freq.minutes
    length : int = 1440
    def download(self): 
        pass
    
    def _load(self):
        # 生成日期序列
        dates = pd.date_range(start='2022-01-01', end='2022-01-02', freq='t')

        # 创建一个数据矩阵
        data = np.random.rand(len(dates), 2)
        # 将时间列和数据矩阵拼接成一个numpy数组
        self.df = pd.DataFrame({'date': dates, 'data1': data[:, 0],'data2': data[:, 1]})
        self.data = self.df.drop('date', axis=1).values        
        return self.data