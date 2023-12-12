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


class NlinearOuputLayer(nn.Module):
    def __init__(self,input_seq_len,num_nodes,out_seq_len,in_dim,tcn_layers,in_channel,dilated_factor,tcn_channel,kernel_set=[2,3,6,7],d0=1) -> None:
        super().__init__()
        
        # self.latent_seq_layer = nn.Linear(input_seq_len, latent_seq)
        self.num_nodes =  num_nodes
        self.pred_len = out_seq_len
        # self.nlinear = NLinear(input_seq_len, out_seq_len, in_dim, True, True)
        self.nlinear = NLinear(input_seq_len, out_seq_len, in_dim, True, True)
        self.mlp_output = nn.Linear(in_dim + in_dim , out_seq_len)
        
    def forward(self, x):
        # (B ,C , N, T)
        
        xs = x[0] # B , N, D
        xt = x[1] # B , T ,D
        B, T, D = xt.size()
        B ,N, D = xs.size()
        xt = self.nlinear(xt) # B O D
        
        
        # 扩展 xs 和 xt 以便于拼接
        xs_expanded = xs.unsqueeze(2)  # 维度变为 [B, N, 1, D]
        xt_expanded = xt.unsqueeze(1)  # 维度变为 [B, 1, O, D]

        # 在拼接维度上重复以匹配对方的维度
        xs_tiled = xs_expanded.repeat(1, 1, self.pred_len, 1)  # [B, N, O, D]
        xt_tiled = xt_expanded.repeat(1, self.num_nodes, 1, 1)  # [B, N, O, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((xs_tiled, xt_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        B, N, O, _ = x_combined.shape
        x_combined = x_combined.reshape(B, N * O, -1)  # [B, N*O, 2D]
        output = self.mlp_output(x_combined)  # [B, N*O, 1]
        output = output.reshape(B, N, O)  # [B, N, O]
        # output = self.mlp_output(xi)  # (B, N, O)
        return output.transpose(1,2)


    
    
    
class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual=False, normalization=True):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.normalization = normalization
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.normalization:
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        if self.normalization:
            x = x + seq_last
        return x # [Batch, Output length, Channel]


# class TCN(nn.Module):
#     def __init__(
#         self, seq_len,num_nodes, in_channels, hidden_channels, out_seq_len=1, out_channels=None, num_layers=5, 
#         d0=1,dilated_factor=2,  dropout=0,kernel_set=[2,3,6,7]
#     ):
#         super().__init__()
#         self.seq_len = seq_len
#         self.dropout = dropout
#         self.act = nn.ReLU()
#         self.num_layers = num_layers

#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         max_kernel_size = max(kernel_set)
#         self.idx = torch.arange(num_nodes)
#         if dilated_factor>1:
#             self.receptive_field = int(1+d0*(max_kernel_size-1)*(dilated_factor**num_layers-1)/(dilated_factor-1))
#         else:
#             self.receptive_field = d0*num_layers*(max_kernel_size-1) + 1
#         # assert self.receptive_field > seq_len  - 1, f"Filter receptive field {self.receptive_field} should be  larger than sequence length {seq_len}"
#         for i in range(1):
#             if dilated_factor>1:
#                 rf_size_i = int(1 + i*(max_kernel_size-1)*(dilated_factor**num_layers-1)/(dilated_factor-1))
#             else:
#                 rf_size_i = i*d0*num_layers*(max_kernel_size-1)+1
#             new_dilation = d0
            
#             for j in range(1,num_layers+1):
#                 if dilated_factor > 1:
#                     rf_size_j = int(rf_size_i + d0*(max_kernel_size-1)*(dilated_factor**j-1)/(dilated_factor-1))
#                 else:
#                     rf_size_j = rf_size_i+d0*j*(max_kernel_size-1)
#                 self.filter_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=new_dilation))
#                 self.gate_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=new_dilation))
                
                
#                 self.residual_convs.append(nn.Conv2d(in_channels=in_channels,
#                                                 out_channels=in_channels,
#                                                 kernel_size=(1, 1)))

#                 if self.seq_len>self.receptive_field:
#                     self.skip_convs.append(nn.Conv2d(in_channels=in_channels,
#                                                     out_channels=in_channels,
#                                                     kernel_size=(1, self.seq_len-rf_size_j+1)))
#                 else:
#                     self.skip_convs.append(nn.Conv2d(in_channels=in_channels,
#                                                     out_channels=in_channels,
#                                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))


                
                
#                 if self.seq_len>self.receptive_field:
#                     self.norms.append(LayerNorm((hidden_channels, num_nodes, seq_len - rf_size_j + 1),elementwise_affine=True))
#                 else:
#                     self.norms.append(LayerNorm((hidden_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=True))

                
#                 new_dilation *= dilated_factor
                
#         # skip layer
#         if self.seq_len > self.receptive_field:
#             self.skip0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, self.seq_len), bias=True)
#             self.skipE = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, self.seq_len-self.receptive_field+1), bias=True)

#         else:
#             self.skip0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, self.receptive_field), bias=True)
#             self.skipE = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), bias=True)

#         self.end_conv = nn.Conv2d(
#             in_channels=hidden_channels, 
#             out_channels=out_channels if out_channels is not None else hidden_channels, 
#             kernel_size=(1, 1), bias=True
#         )


#     def forward(self, x):
#         """
#         x.shape: (B, Cin, N, T)

#         output.shape: (B, Cout, N, out_seq_len)
#         """
        

#         batch, _, n_nodes, seq_len = x.shape
#         assert seq_len == self.seq_len, f"Sequence length {seq_len} should be {self.seq_len}"
        
#         if seq_len < self.receptive_field:
#             x = nn.functional.pad(x, (self.receptive_field - seq_len, 0, 0, 0))

#         origin = x
#         skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
#         for i in range(self.num_layers):
#             seq_last = x[:,:,:,-1:].detach()
#             x = x - seq_last

            
            
#             residual = x
#             filter = self.filter_convs[i](x)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](x)
#             gate = torch.sigmoid(gate)
#             x = filter * gate
#             x = F.dropout(x, self.dropout, training=self.training)
#             s = x # (n , 1 , m , p)
#             s = self.skip_convs[i](s) 
#             skip = s + skip

#             x = self.residual_convs[i](x)
#             x = x + residual[:, :, :, -x.size(3):]

#             x = self.norms[i](x,self.idx)
            
            
#             x = x + seq_last
        
#         skip = self.skipE(x) + skip
#         x = F.elu(skip)
            
#         x = F.elu(self.end_conv(x))
#         return x
    
    
    