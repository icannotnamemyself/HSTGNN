
import pytest
import os
from torch_timeseries.datasets import ExchangeRate
from torch_timeseries.data.scaler import StandarScaler
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader
def test_dataloader():
    dataloader = ChunkSequenceTimefeatureDataLoader(
        ExchangeRate(),
        scaler=StandarScaler()
    )
    for  _ in dataloader.train_loader:
        continue
    

def test_exchange_rate():
    dataset = ExchangeRate(root='./data')
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features

    
