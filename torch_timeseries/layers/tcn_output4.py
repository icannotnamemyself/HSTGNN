import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.tcnv4 import TCN


class TCNOuputLayer(nn.Module):
    def __init__(self,input_seq_len,num_nodes,out_seq_len,tcn_layers,dilated_factor,in_channel,tcn_channel) -> None:
        super().__init__()
        self.channel_layer = nn.Conv2d(in_channel, tcn_channel, (1, 1))
        self.tcn = TCN(
                    input_seq_len,num_nodes, tcn_channel, tcn_channel,
                    out_seq_len=out_seq_len, num_layers=tcn_layers,
                    dilated_factor=dilated_factor
                )
        self.end_layer = nn.Conv2d(tcn_channel, out_seq_len, (1, 1))
        self.act = nn.ELU()
        
    def forward(self, x):
        output = self.channel_layer(x)  # (B ,C , N, latent_seq)
        output = self.act(self.tcn(output))  # (B, C, N, out_len)
        output = self.end_layer(output).squeeze(3)  # (B, out_len, N)
        return output
