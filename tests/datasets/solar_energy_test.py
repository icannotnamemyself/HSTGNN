
import pytest
import os
from torch_timeseries.data.scaler import StandarScaler
from torch_timeseries.datasets import SolarEnergy
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader


def test_solar_energy():
    dataset = SolarEnergy(root='./data')
    assert os.path.exists("./data/solar_AL/solar_AL.txt") is True
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features


def test_dataloader():
    dataloader = ChunkSequenceTimefeatureDataLoader(
        SolarEnergy(),
        scaler=StandarScaler()
    )
    
    
    for  _ in dataloader.train_loader:
        continue