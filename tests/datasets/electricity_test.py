
import pytest
import os
from torch_timeseries.datasets import Electricity , ElectricityV2
from torch_timeseries.datasets.wrapper import SingleStepWrapper, MultiStepWrapper

def test_exchange_rate():
    electricity = Electricity(root='./data')
    df = electricity.raw_df()
    assert os.path.exists("./data/electricity/raw/electricity.txt.gz") is True
    assert df.shape[0] == 26304 and df.shape[1] == 321

def test_exchange_ratev2():
    electricity = ElectricityV2(root='./data')
    assert os.path.exists(f"./data/{electricity.name}/raw/electricity.txt.gz") is True
    
    assert electricity.data.shape[0] == 26304 and electricity.data.shape[1] == 321
    
    to_loader_wrapper = SingleStepWrapper(electricity, window=155, horizon=3)
    assert len(to_loader_wrapper[0][0]) == 155 
    


    
    
    
    

