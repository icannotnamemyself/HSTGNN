
from turtle import forward
from torch import Tensor
import torch.nn as nn
from torch_timeseries.nn.dialted_inception import DilatedInception
from torch_timeseries.nn.timeseries_startconv import TimeSeriesStartConv
import torch.nn.functional as F
import torch
from experiments.net import Net


def test_net():
    d = torch.rand(64, 1, 50,50)
    n = Net(input_node=50,seq_len=50,in_dim=1, embed_dim=10, middle_channel=32, seq_out=1,dilation_exponential=2, layers=5)
    aa:Tensor =  n(d) # -> ( b, seq_out_len ,n ,1)
    aa = aa.squeeze(dim=3) # -> ( b, seq_out_len, n)
    aa = aa.transpose(1,2)  # -> ( b, n, seq_out_len)
