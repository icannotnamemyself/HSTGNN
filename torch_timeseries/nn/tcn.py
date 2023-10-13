import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.dilated_inception import DilatedInception

class TCN(nn.Module):
    def __init__(
        self, seq_len, in_channels, hidden_channels, out_seq_len=1, out_channels=None, num_layers=2, 
        dilated_factor=2, dropout=0
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dropout = dropout
        self.act = nn.ReLU()
        self.num_layers = num_layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.filter_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=dilated_factor))
        self.gate_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=dilated_factor))
        for _ in range(num_layers - 2):
            self.filter_convs.append(DilatedInception(hidden_channels, hidden_channels, dilation_factor=dilated_factor))
            self.gate_convs.append(DilatedInception(hidden_channels, hidden_channels, dilation_factor=dilated_factor))
        
        if out_channels is not None:
            self.filter_convs.append(DilatedInception(hidden_channels, out_channels, dilation_factor=dilated_factor))
            self.gate_convs.append(DilatedInception(hidden_channels, out_channels, dilation_factor=dilated_factor))
        else:
            self.filter_convs.append(DilatedInception(hidden_channels, hidden_channels, dilation_factor=dilated_factor))
            self.gate_convs.append(DilatedInception(hidden_channels, hidden_channels, dilation_factor=dilated_factor))
    
        max_kernal_size = 7
        receptive_field = num_layers * dilated_factor * (max_kernal_size - 1) + 1
        self.min_input_len = receptive_field + out_seq_len - 1

        end_kernel_size = 1
        if seq_len > self.min_input_len:
            end_kernel_size = seq_len - self.min_input_len + 1
        self.end_conv = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=out_channels if out_channels is not None else hidden_channels, 
            kernel_size=(1, end_kernel_size), bias=True
        )


    def forward(self, x):
        """
        x.shape: (B, Cin, N, T)

        output.shape: (B, Cout, N, out_seq_len)
        """
        batch, _, n_nodes, seq_len = x.shape
        assert seq_len == self.seq_len
        
        if seq_len < self.min_input_len:
            x = nn.functional.pad(x, (self.min_input_len - seq_len, 0, 0, 0))

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