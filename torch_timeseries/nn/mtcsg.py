from numpy import inf
import torch.nn as nn
import torch
from torch_timeseries.nn.dil_encoder import DILEncoder
from torch_timeseries.nn.cnn_decoder import CNNDecoder
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from loguru import logger


import pdb

def matrix_poly(matrix, d):
    x = torch.eye(d).double()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


class GCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        
    def forward(self, x):
        pass


class MTCSG(nn.Module):
    
    
    def __init__(self, node_num:int, seq_len:int, in_dim=1, causal_aggr_layers=1,seq_out=1, middle_channel=64) -> None:
        super().__init__()
        
        self.node_num = node_num
        self.causal_aggr_layers = causal_aggr_layers
        
        self.encoder:nn.Module = DILEncoder(node_num, seq_len,in_dim=in_dim, middle_channel=middle_channel,dilation_exponential=2, layers=5)
        
        self.weighted_A = nn.Parameter(torch.randn(self.node_num,  self.node_num))
        
        self.graph_conv = GCNConv(middle_channel, middle_channel)
        
        self.decoder:nn.Module = CNNDecoder(middle_channel=middle_channel, out_dim=seq_out)
        
        
        self.h_A = inf
        
    def forward(self, x):
        
        x = self.encoder(x) # ( b, middle_channel, node_num, 1)
        
        x =x.squeeze()
        x = x.transpose(1, 2)
        
        # graph convolution
        # conv_A = self.weighted_A + torch.eye(self.node_num)
        edge_indices, edge_attributes = dense_to_sparse(self.weighted_A)
        for i in range(self.causal_aggr_layers):
            x = self.graph_conv(x, edge_indices, edge_attributes.sigmoid())
        # x (b, n, channel)
        
        
        x = x.transpose(1,2)
        x = torch.unsqueeze(x, 3)
        
       
        x = self.decoder(x) #( b, middle_channel, node_num  , 1) -> ( b, node_num, 1) 
        
        return  x
        
        
    




