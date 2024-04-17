


import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from class_resolver.contrib.torch import activation_resolver
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import FAConv,HeteroConv
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear



class GraphSage(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, n_layers, 
        out_channels=None, dropout=0, norm=None,
        act='relu', act_first=False, eps=0.1, **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = n_layers

        self.dropout = dropout
        self.act = activation_resolver.make(act)
        self.act_first = act_first
        self.eps = eps

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = nn.ModuleList()
        self.convs.append(
            self.init_conv(in_channels, hidden_channels, **kwargs))
        for _ in range(n_layers - 2):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        if out_channels is not None:
            self.convs.append(
                self.init_conv(hidden_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        
        self.norms = None
        if norm is not None:
            self.norms = nn.ModuleList()
            for _ in range(n_layers - 1):
                self.norms.append(copy.deepcopy(norm))
    
    def init_conv(self, in_channels, out_channels, **kwargs):
        return EGraphSage(in_channels, out_channels, activation=self.act, **kwargs)

    def forward(self, x , edge_index, edge_attr):
        # if len(edge_attr.shape) == 2:
        #     edge_attr = edge_attr.unsqueeze(-1)  # (B, E) -> (B, E, 1)
        origin = x
        for i in range(self.num_layers):
            xs = list()
            for bi in range(x.shape[0]):
                xs.append(self.convs[i](x[bi], edge_attr[bi], edge_index[bi]))
            x = torch.stack(xs)
            if i == self.num_layers - 1:
                break
            
            if self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if not self.act_first:
                x = self.act(x)
            
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = origin + x * self.eps
        return x



class EGraphSage(MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels,
                 edge_channels=1, activation='elu', edge_mode=1,
                 normalize_emb=False, aggr='add'):
        super(EGraphSage, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.edge_mode = edge_mode
        self.act = activation_resolver.make(activation)

        if edge_mode == 0:
            self.message_lin = nn.Linear(in_channels, out_channels)
            self.attention_lin = nn.Linear(2*in_channels+edge_channels, 1)
        elif edge_mode == 1:
            self.message_lin = nn.Linear(in_channels+edge_channels, out_channels)
        elif edge_mode == 2:
            self.message_lin = nn.Linear(2*in_channels+edge_channels, out_channels)
        elif edge_mode == 3:
            self.message_lin = nn.Sequential(
                    nn.Linear(2*in_channels+edge_channels, out_channels),
                    self.act,
                    nn.Linear(out_channels, out_channels),
                    )
        elif edge_mode == 4:
            self.message_lin = nn.Linear(in_channels, out_channels*edge_channels)
        elif edge_mode == 5:
            self.message_lin = nn.Linear(2*in_channels, out_channels*edge_channels)

        self.agg_lin = nn.Linear(in_channels+out_channels, out_channels)

        self.message_activation = self.act
        self.update_activation = self.act
        self.normalize_emb = normalize_emb

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]
        if self.edge_mode == 0:
            attention = self.attention_lin(torch.cat((x_i,x_j, edge_attr),dim=-1))
            m_j = attention * self.message_activation(self.message_lin(x_j))
        elif self.edge_mode == 1:
            edge_attr = edge_attr.view(-1, 1)
            m_j = torch.cat((x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 2 or self.edge_mode == 3:
            m_j = torch.cat((x_i,x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 4:
            E = x_j.shape[0]
            w = self.message_lin(x_j)
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        elif self.edge_mode == 5:
            E = x_j.shape[0]
            w = self.message_lin(torch.cat((x_i,x_j),dim=-1))
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        return m_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x),dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out



class GCN(GraphSage):
    def __init__(
        self, in_channels, hidden_channels, n_layers, out_channels=None,
        dropout=0, norm=None, act='relu', act_first=False, eps=0.1, **kwargs
    ):
        super().__init__(in_channels, hidden_channels, n_layers, out_channels, dropout, norm, act, act_first, eps, **kwargs)
    
    def init_conv(self, in_channels, out_channels, dropout=0, **kwargs):
        return FAConv(in_channels, dropout=dropout, eps=self.eps, **kwargs)
        # return FAConv(in_channels, out_channels, **kwargs)
    
    def forward(self, x, edge_index, edge_attr=None):
        # x: B * (N+T) * C
        # edge_index: B,2,2*(N*T)
        # edge_attr: B*E or B * (N * T )
        # if len(edge_attr.shape) == 2:
        #     edge_attr = edge_attr.unsqueeze(-1)  # (B, E) -> (B, E, 1)
        origin = x
        for i in range(self.num_layers):
            xs = list()
            for bi in range(x.shape[0]):
                xs.append(self.convs[i](x[bi], origin[bi], edge_index[bi]))
            x = torch.stack(xs)
            if i == self.num_layers - 1:
                break
            
            if self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if not self.act_first:
                x = self.act(x)
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
