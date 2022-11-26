import pytest
import socket
import pyChainedProxy as socks

import numpy as np
import pandas as pd
from torch_timeseries.datasets.dataset import TimeSeriesDataset



class DummyDataset(TimeSeriesDataset):
    name: str = 'dummy'
    num_features:int = 100
    sample_rate:int = 1
    def download(self): 
        pass
    
    def _load(self):
        l = []
        for i in range(0, 10000):
            l.append([i]* self.num_features)
        return np.array(l)

    def _process(self):
        return super()._process()

@pytest.fixture(scope='session')
def dummy_dataset():
    return DummyDataset('./data')

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


