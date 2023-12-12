import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.dilated_inception import DilatedInception
from torch_timeseries.nn.layer import LayerNorm


class TCNOuputLayer(nn.Module):
    def __init__(self,input_seq_len,num_nodes,out_seq_len,tcn_layers,in_channel,dilated_factor,tcn_channel,kernel_set=[2,3,6,7],d0=1) -> None:
        super().__init__()
        
        # self.latent_seq_layer = nn.Linear(input_seq_len, latent_seq)
        self.channel_layer = nn.Conv2d(in_channel, tcn_channel, (1, 1))
        self.tcn = TCN(
                    input_seq_len,num_nodes, tcn_channel, tcn_channel,
                    out_seq_len=out_seq_len, num_layers=tcn_layers,
                    dilated_factor=dilated_factor, d0=d0,kernel_set=kernel_set
                )
        self.end_layer = nn.Conv2d(tcn_channel, out_seq_len, (1, 1))
        self.act = nn.ELU()
        
    def forward(self, x):
        # output = self.act(self.latent_seq_layer(x))# (B ,C , N, latent_seq)
        output = self.channel_layer(x)  # (B ,C , N, T)
        output = self.act(self.tcn(output))  # (B, C, N, out_len)
        output = self.end_layer(output).squeeze(3)  # (B, out_len, N)
        return output




class TCN(nn.Module):
    def __init__(
        self, seq_len,num_nodes, in_channels, hidden_channels, out_seq_len=1, out_channels=None, num_layers=5, 
        d0=1,dilated_factor=2,  dropout=0,kernel_set=[2,3,6,7]
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dropout = dropout
        self.act = nn.ReLU()
        self.num_layers = num_layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        max_kernel_size = max(kernel_set)
        self.idx = torch.arange(num_nodes)
        if dilated_factor>1:
            self.receptive_field = int(1+d0*(max_kernel_size-1)*(dilated_factor**num_layers-1)/(dilated_factor-1))
        else:
            self.receptive_field = d0*num_layers*(max_kernel_size-1) + 1
        # assert self.receptive_field > seq_len  - 1, f"Filter receptive field {self.receptive_field} should be  larger than sequence length {seq_len}"
        for i in range(1):
            if dilated_factor>1:
                rf_size_i = int(1 + i*(max_kernel_size-1)*(dilated_factor**num_layers-1)/(dilated_factor-1))
            else:
                rf_size_i = i*d0*num_layers*(max_kernel_size-1)+1
            new_dilation = d0
            
            for j in range(1,num_layers+1):
                if dilated_factor > 1:
                    rf_size_j = int(rf_size_i + d0*(max_kernel_size-1)*(dilated_factor**j-1)/(dilated_factor-1))
                else:
                    rf_size_j = rf_size_i+d0*j*(max_kernel_size-1)
                self.filter_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=new_dilation))
                
                
                self.residual_convs.append(nn.Conv2d(in_channels=in_channels,
                                                out_channels=in_channels,
                                                kernel_size=(1, 1)))

                if self.seq_len>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=(1, self.seq_len-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))


                
                
                if self.seq_len>self.receptive_field:
                    self.norms.append(LayerNorm((hidden_channels, num_nodes, seq_len - rf_size_j + 1),elementwise_affine=True))
                else:
                    self.norms.append(LayerNorm((hidden_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=True))

                
                new_dilation *= dilated_factor
                
        # skip layer
        if self.seq_len > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, self.seq_len), bias=True)
            self.skipE = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, self.seq_len-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), bias=True)

        self.end_conv = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=out_channels if out_channels is not None else hidden_channels, 
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
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        for i in range(self.num_layers):
            seq_last = x[:,:,:,-1:].detach()
            x = x - seq_last

            
            
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x # (n , 1 , m , p)
            s = self.skip_convs[i](s) 
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.norms[i](x,self.idx)
            
            
            x = x + seq_last
        
        skip = self.skipE(x) + skip
        x = F.elu(skip)
            
        x = F.elu(self.end_conv(x))
        return x