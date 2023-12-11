import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.tcnv6 import TCN


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
