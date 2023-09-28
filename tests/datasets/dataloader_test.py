from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader

def test_dataloader():
    dataloader = ChunkSequenceTimefeatureDataLoader(
        ETTh1(),
        scaler=StandarScaler(),
        steps=2
    )
    
    
    
    for x, y, x_enc_date, y_enc_date in dataloader.train_loader:
        continue