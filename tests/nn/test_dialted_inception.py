from ctypes import c_int
import pytest
import torch
from torch_timeseries.nn.dilated_inception import DilatedInception


def test_dialted_inception():
    data = torch.randn(size=(64,3,8,144))
    input = data
    dil = DilatedInception(cin=3, cout=16)  # torch.Size([64, 16, 8, 132])
    print(dil(input).size())
    
    
    




