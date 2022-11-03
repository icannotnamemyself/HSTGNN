from ctypes import c_int
import pytest
import torch
from torch_timeseries.nn.timeseries_startconv import TimeSeriesStartConv


def test_start_conv():
    data = torch.randn(size=(64,3,8,144))
    input = data
    start_conv = TimeSeriesStartConv(channel_in=3, channel_out=16)  # torch.Size([64, 16, 8, 144])
    print(start_conv(input).size())
    
    
    




