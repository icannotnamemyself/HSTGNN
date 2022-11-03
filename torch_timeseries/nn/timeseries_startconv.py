import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F



class TimeSeriesStartConv(nn.Module):
    def __init__(self, channel_in, channel_out) -> None:
        super().__init__()
        self.start_conv = nn.Conv2d(in_channels=channel_in,out_channels=channel_out,kernel_size=(1, 1))
    def forward(self, input):
        return self.start_conv(input)