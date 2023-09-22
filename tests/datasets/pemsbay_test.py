from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader

import os
def test_metr_la():
    dataset = PEMS_BAY(root='./data')
    assert os.path.exists("./data/PEMS-BAY/pems-bay.h5") is True
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features

def test_dataloader():
    dataloader = ChunkSequenceTimefeatureDataLoader(
        PEMS_BAY(),
        scaler=StandarScaler()
    )
    
    
    for  _ in dataloader.train_loader:
        continue