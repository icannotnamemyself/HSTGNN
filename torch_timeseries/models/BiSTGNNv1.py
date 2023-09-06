from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt

from math import ceil

import torch

def biadj_to_adjacency_matrix(s_to_ajd, t_to_s_adj):
    # input: (NxT)
    #output : (N+T) x (N+T)
    N, T = s_to_ajd.size()
    
    # 创建完整的邻接矩阵
    adjacency_matrix = torch.zeros(N+T, N+T)
    
    # 将二部图的NxT矩阵赋值给邻接矩阵的相应位置
    adjacency_matrix[:N, N:] = s_to_ajd
    adjacency_matrix[N:, :N] = t_to_s_adj  # 转置矩阵
       
    return adjacency_matrix




class BiSTGN(nn.Module):
    def __init__(self, seq_len, num_nodes,temporal_embed_dim, rank, node_static_embed_dim=32, latent_dim=256):
        super(SpatialEncoder, self).__init__()
        
        
        self.num_nodes = num_nodes
        self.static_embed_dim = node_static_embed_dim
        self.latent_dim = latent_dim
        
        
        self.spatial_encoder = SpatialEncoder(seq_len, num_nodes, static_embed_dim=node_static_embed_dim, latent_dim=latent_dim)        
        self.temporal_encoder = TemporalEncoder(seq_len, num_nodes, temporal_embed_dim=temporal_embed_dim, latent_dim=latent_dim)        
        
        
        self.graph_constructor = STGraphConstructor(latent_dim, rank=rank)
        
    def forward(self, x, x_enc_mark):
        """
            in :  (B, N, T) 
            out:  (B, N, latent_dim)
        """
        
        
        Xs = self.spatial_encoder(x)
        Xt = self.temporal_encoder(x, x_enc_mark)
        
        
        s_to_ajd, t_to_s_adj = self.graph_constructor(Xs, Xt) # spatial and temporal adjecent matrix
        
        A = biadj_to_adjacency_matrix(s_to_ajd, t_to_s_adj)
        
        
        
        
        
        
        
        





class SpatialEncoder(nn.Module):
    def __init__(self, seq_len, num_nodes, static_embed_dim=32, latent_dim=256):
        super(SpatialEncoder, self).__init__()
        
        
        self.num_nodes = num_nodes
        self.static_embed_dim = static_embed_dim
        self.latent_dim = latent_dim
        
        self.static_node_embedding = nn.Embedding(num_nodes, static_embed_dim)
        self.spatial_projection = nn.Linear(seq_len+static_embed_dim, latent_dim)
        
    def forward(self, x):
        """
            in :  (B, N, T) 
            out:  (B, N, latent_dim)
        """
        
        
        # 获取输入张量x的形状
        B, N, T = x.size()
        assert N == self.num_nodes
        
        idx = torch.tensor(range(self.num_nodes))
        static_embeding = self.static_node_embedding(idx).expand(B, N, -1)
        
        # 将static_embedding和x沿最后一个维度连接，得到(B, N, T+static_embed_dim)的新张量
        x = torch.cat((x, static_embeding), dim=2)
        
        # 对combined_x进行进一步处理和计算
        spatial_encode = self.spatial_projection(x)
        
        return spatial_encode



class TemporalEncoder(nn.Module):
    def __init__(self,num_nodes, temporal_embed_dim, latent_dim=256):
        super(TemporalEncoder, self).__init__()
        
        
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        
        self.temporal_projection = nn.Linear(num_nodes+temporal_embed_dim, latent_dim)
        
    def forward(self, x, x_enc_mark):
        """
            x :  (B, N, T)
            x_enc_mark : (B, T, D_t)
            
            
            out:  (B, T, latent_dim)
        """
        
        
        # 获取输入张量x的形状
        B, N, T = x.size()
        assert N == self.num_nodes
        
        x = torch.cat((x.transpose(1,2), x_enc_mark), dim=2)  # (B, T, D_t + N)
        
        # 对combined_x进行进一步处理和计算
        temporal_encode = self.temporal_projection(x)
        
        return temporal_encode





class STGraphConstructor(nn.Module):
    def __init__(self, latent_dim, rank=128):
        super(STGraphConstructor, self).__init__()
        self.temporal_att_weight = nn.Parameter(torch.randn(latent_dim,rank))
        self.spatial_att_weight = nn.Parameter(torch.randn(latent_dim,rank))
        
        self.sp = nn.Parameter(torch.randn(rank , 1))
        self.tp = nn.Parameter(torch.randn(rank , 1))
        
    def forward(self, spatial_nodes, temporal_nodes) -> Tuple[torch.Tensor, torch.Tensor]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        B, N, D = spatial_nodes.size()
        _, T, D = spatial_nodes.size()
        
        t_to_s_att = (torch.ones(T, 1) @ (self.tp.T)  * (temporal_nodes@ self.temporal_att_weight)) @ (self.spatial_att_weight.T @ spatial_nodes.transpose(1,2))
        s_to_t_att = (torch.ones(N, 1) @ (self.sp.T)  * (spatial_nodes@ self.spatial_att_weight)) @ (self.temporal_att_weight.T @ temporal_nodes.transpose(1,2))
        return nn.Sigmoid(s_to_t_att), nn.Sigmoid(t_to_s_att)
        
    





# https://github.com/bdy9527/FAGCN/tree/main
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np


class FALayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class FAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)