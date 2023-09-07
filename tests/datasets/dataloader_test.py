from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader

def test_dataloader():
    dataloader = ChunkSequenceTimefeatureDataLoader(
        ETTh1(),
        scaler=StandarScaler()
    )
    
    