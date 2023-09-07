
import pytest
import os
from torch_timeseries.datasets import Electricity , ILI, Weather
from torch_timeseries.datasets.wrapper import SingleStepWrapper, MultiStepWrapper
from torch_timeseries.data.scaler import StandarScaler
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader
def test_dataloader():
    dataloader = ChunkSequenceTimefeatureDataLoader(
        Electricity(),
        scaler=StandarScaler()
    )
    for  _ in dataloader.train_loader:
        continue
    dataloader = ChunkSequenceTimefeatureDataLoader(
        Weather(),
        scaler=StandarScaler()
    )
    
    
    for  _ in dataloader.train_loader:
        continue
    dataloader = ChunkSequenceTimefeatureDataLoader(
        ILI('./data'),
        scaler=StandarScaler()
    )
    for  _ in dataloader.train_loader:
        continue

def test_exchange_rate():
    dataset = Electricity(root='./data')
    assert os.path.exists("./data/electricity/electricity.csv") is True
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features



def test_ili():
    dataset = ILI(root='./data')
    assert os.path.exists("./data/ILI/illness/national_illness.csv") is True
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features



def test_weather():
    dataset = Weather(root='./data')
    assert os.path.exists("./data/weather/weather.csv") is True
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features


    
    
    
    

