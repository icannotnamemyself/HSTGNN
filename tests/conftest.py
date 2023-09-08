import pytest
import socket
import pyChainedProxy as socks
import torch
import numpy as np
import pandas as pd
from torch_timeseries.datasets.dataset import Freq, TimeSeriesDataset



class DummyDataset(TimeSeriesDataset):
    name: str = 'dummy'
    num_features:int = 8
    sample_rate:int = 1
    length : int= 1000
    def download(self): 
        pass
    
    def _load(self):
        # 生成日期序列
        dates = pd.date_range(start='2022-01-01', end='2022-01-02', freq='t')

        # 创建一个数据矩阵
        data = np.random.rand(len(dates), 2)

        # 将时间列和数据矩阵拼接成一个numpy数组
        result = np.concatenate([dates[:, np.newaxis], data], axis=1)

        # 创建DataFrame，指定列名
        self.df = pd.DataFrame(result, columns=['date', 'data1', 'data2'])
        self.data = self.df.drop('date').values        
        return self.data


class DummyDatasetWithTime(TimeSeriesDataset):
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
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop('date', axis=1).values        
        return self.data


@pytest.fixture(scope='session')
def dummy_dataset():
    return DummyDataset('./data')

@pytest.fixture(scope='session')
def dummy_dataset_time():
    return DummyDatasetWithTime('./data')



@pytest.fixture(scope='session')
def batch_x():
    batch_size = 64
    seq_len = 128
    input_dim = 256
    timeseries = torch.rand(batch_size, seq_len, input_dim)
    return timeseries


@pytest.fixture(scope='session')
def use_proxy():
    
    import socket
    import pyChainedProxy as socks

    chain = [
    'socks5://127.0.0.1:7890', 
    ]
    socks.setdefaultproxy() # Clear the default chain
    #adding hops with proxies
    for hop in chain:
        socks.adddefaultproxy(*socks.parseproxy(hop))

    # Configure alternate routes (No proxy for localhost)
    socks.setproxy('localhost', socks.PROXY_TYPE_NONE)
    socks.setproxy('127.0.0.1', socks.PROXY_TYPE_NONE)

    # Monkey Patching whole socket class (everything will be proxified)
    rawsocket = socket.socket
    socket.socket = socks.socksocket


