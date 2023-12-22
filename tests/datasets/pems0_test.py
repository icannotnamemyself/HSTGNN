from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader

import os
def test_pems_03():
    dataset = PEMS03(root='./data')
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features
    dataloader = ChunkSequenceTimefeatureDataLoader(
        dataset,
        scaler=StandarScaler()
    )
    
    
    for  _ in dataloader.train_loader:
        continue

def test_pems_04():
    dataset = PEMS04(root='./data')
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features
    dataloader = ChunkSequenceTimefeatureDataLoader(
        dataset,
        scaler=StandarScaler()
    )
    
    
    for  _ in dataloader.train_loader:
        continue

def test_pems_07():
    dataset = PEMS07(root='./data')
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features
    dataloader = ChunkSequenceTimefeatureDataLoader(
        dataset,
        scaler=StandarScaler()
    )
    
    
    for  _ in dataloader.train_loader:
        continue

def test_pems_08():
    dataset = PEMS08(root='./data')
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features
    dataloader = ChunkSequenceTimefeatureDataLoader(
        dataset,
        scaler=StandarScaler()
    )
    
    for  _ in dataloader.train_loader:
        continue
