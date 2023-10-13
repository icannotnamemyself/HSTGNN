from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader

import os
def test_pems_d7():
    dataset = PeMS_D7(root='./data')
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features

def test_dataloader():
    dataloader = ChunkSequenceTimefeatureDataLoader(
        PeMS_D7(),
        scaler=StandarScaler()
    )
    
    
    for  _ in dataloader.train_loader:
        continue