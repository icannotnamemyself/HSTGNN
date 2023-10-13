import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.dilated_inception import DilatedInception

class TCN(nn.Module):
    def __init__(
        self, seq_len, channels, out_seq_len=1, num_layers=3, 
        dilated_factor=2, dropout=0,kernel_set=[2,3,6,7]
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dropout = dropout
        self.act = nn.ReLU()
        self.num_layers = num_layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        max_kernel_size = max(kernel_set)
        

        self.receptive_field = num_layers * dilated_factor * (max_kernel_size - 1) + 1
        assert self.receptive_field > seq_len  - 1, f"Filter receptive field {self.receptive_field} should be  larger than sequence length {seq_len}"
        for j in range(1,num_layers+1):
            self.filter_convs.append(DilatedInception(channels, channels, dilation_factor=dilated_factor,kernel_set=kernel_set))
            self.gate_convs.append(DilatedInception(channels, channels, dilation_factor=dilated_factor,kernel_set=kernel_set))

        self.end_conv = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=(1, 1), bias=True
        )


    def forward(self, x):
        """
        x.shape: (B, Cin, N, T)

        output.shape: (B, Cout, N, out_seq_len)
        """
        batch, _, n_nodes, seq_len = x.shape
        assert seq_len == self.seq_len, f"Sequence length {seq_len} should be {self.seq_len}"
        
        if seq_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - seq_len, 0, 0, 0))

        origin = x
        for i in range(self.num_layers):
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            x = x + origin[:, :, :, -x.shape[3]:]
        
        x = self.end_conv(x)
        
        return x