

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from class_resolver.contrib.torch import activation_resolver
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import FAConv,HeteroConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear

from torch_timeseries.utils.norm import hetero_directed_norm
# from torch_timeseries.layers.graphsage import MyGraphSage



class HeteroFASTGCN(nn.Module):
    def __init__(
        self,node_num,seq_len, in_channels, hidden_channels, n_layers, out_channels=None,
        dropout=0, norm=None, act='relu',n_first=True, act_first=False, eps=0.9, **kwargs
    ):
        
        self.node_num =node_num
        self.seq_len = seq_len
        self.n_first = n_first

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
            
        assert n_layers >= 2 , "intra and inter conv layers must greater than or equals to 2 "

        self.convs = nn.ModuleList()
        
        self.convs.append(self.init_intra_conv(in_channels, hidden_channels))

        for i in range(n_layers - 2):
            if i % 2 == 1: # intra_conv
                self.convs.append(self.init_intra_conv(in_channels, hidden_channels))
            else: # inter conv
                self.convs.append(self.init_inter_conv(hidden_channels, hidden_channels))
          
            
        if n_layers % 2 == 1: # intra_conv
            self.convs.append(self.init_intra_conv(in_channels, out_channels))
        else: # inter conv
            self.convs.append(self.init_inter_conv(hidden_channels, out_channels))

        self.norms = None
        if norm is not None:
            self.norms = nn.ModuleList()
            for _ in range(n_layers - 1):
                self.norms.append(copy.deepcopy(norm))
            
    def init_intra_conv(self, in_channels, out_channels, **kwargs):
        # print("init_intra")
        intrast_homo_conv = HeteroConv({
            ('s', 's2s', 's'): SSConv(in_channels, out_channels, eps=self.eps),
            ('t', 't2t', 't'): TTConv(in_channels, out_channels, eps=self.eps),
        }, aggr='sum')
        return intrast_homo_conv

    def init_inter_conv(self, in_channels, out_channels, **kwargs):
        # print("init_inter")
        interst_biparte_conv = HeteroConv({
            ('s', 's2t', 't'): STConv(in_channels, out_channels, eps=self.eps),
            ('t', 't2s', 's'): TSConv(in_channels, out_channels, eps=self.eps),
        }, aggr='sum')
        return interst_biparte_conv
        # return FAConv(in_channels, out_channels, **kwargs)

    
    def forward(self, x, edge_index, edge_attr=None):
        # x: B * (N+T) * C
        # edge_index: B,2,2*(N*T)
        # edge_attr: B*E or B * (N * T )

        for i in range(self.num_layers):
            xs = list()
            for bi in range(x.shape[0]):

                x_dict = {
                    's': x[bi][:self.node_num,:],
                    't': x[bi][self.node_num:,:]
                }
                edge_index_bi = edge_index[bi]
                if i % 2 == 0: # intra
                    edge_nn = edge_index_bi[:, (edge_index_bi[0] < self.node_num) & (edge_index_bi[1] < self.node_num)]
                    edge_tt = edge_index_bi[:, (edge_index_bi[0] >=self.node_num ) & (edge_index_bi[1]  >=self.node_num)]
                    
                    # set tt edge start index to 0
                    edge_tt = edge_tt - self.node_num
                    edge_index_dict = {
                        ('s', 's2s', 's'): edge_nn,
                        ('t', 't2t', 't'): edge_tt,
                    }
                    out_dict = self.convs[i](x_dict,edge_index_dict )
                    xi = x[bi] + torch.concat([out_dict['s'], out_dict['t']], dim=0)
                    # print("prop intra")
                    
                else: # inter
                    edge_nt = edge_index_bi[:, (edge_index_bi[0] < self.node_num) & (edge_index_bi[1] >= self.node_num)]
                    edge_tn = edge_index_bi[:, (edge_index_bi[0] >= self.node_num) & (edge_index_bi[1] < self.node_num)]        
                    edge_index_dict = {
                        ('s', 's2t', 't'): edge_nt,
                        ('t', 't2s', 's'): edge_tn,
                    }
                    # print("prop inter")
                    
                    out_dict = self.convs[i](x_dict,edge_index_dict )
                    
                    xi = x[bi] + out_dict['s'] + out_dict['t']
                # combining spatial and temporal mixed information
                
                # xi = self.convs[i](x[bi], edge_index[bi])
                xs.append(xi)
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


class TSConv(MessagePassing):
    # convolution for relation < t -> s >
    def __init__(self, in_channels, out_channels, eps=0.9):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.eps = eps
        self.att_g = Linear(2*in_channels,1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.att_g.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N+T, in_channels]
        # edge_index has shape [2, E]
        # edge_index has shape [E, weight_dim]
        
        xt = x[0]
        xs = x[1]
        
        x = torch.concat([xs, xt], dim=0)
        
        edge_weight = hetero_directed_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim), dtype=x.dtype)

        t2s_info = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        return t2s_info

    def message(self,x_i, x_j, edge_weight):
        # x_j has shape [|E|, out_channels] , The first n edges are edges of  spatial nodes
        # x_j denotes a lifted tensor, which contains the source node features of each edge, source_node(如果flow 是 source_to_target)
        # 要从从有向图的角度来解释 edge_index 有几个，就有几个x_
        
        # 对 st , ts分别定义
        
        alpha_i_j = self.att_g(torch.concat([x_i, x_j], axis=1)).tanh().squeeze(-1) # ( |E|, )    
        
        return x_j *( alpha_i_j * edge_weight ).view(-1,1)


class STConv(MessagePassing):
    # convolution for relation < s -> t >
    def __init__(self, in_channels, out_channels, eps=0.9):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.eps = eps
        self.att_g = Linear(2*in_channels,1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.att_g.reset_parameters()
        # self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N+T, in_channels]
        # edge_index has shape [2, E]
        # edge_index has shape [E, weight_dim]
        xs = x[0]
        xt = x[1]
        
        x = torch.concat([xs, xt], dim=0)
        
        edge_weight = hetero_directed_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim), dtype=x.dtype)
        s2t_info = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return s2t_info

    def message(self,x_i, x_j, edge_weight):
        # x_j has shape [|E|, out_channels] , The first n edges are edges of  spatial nodes
        # x_j denotes a lifted tensor, which contains the source node features of each edge, source_node(如果flow 是 source_to_target)
        # 要从从有向图的角度来解释 edge_index 有几个，就有几个x_
        
        # 对 st , ts分别定义
        alpha_i_j = self.att_g(torch.concat([x_i, x_j], axis=1)).tanh().squeeze(-1) # ( |E|, )    
        
        return x_j *( alpha_i_j * edge_weight ).view(-1,1)



class SSConv(MessagePassing):
    # convolution for relation < s -> t >
    def __init__(self, in_channels, out_channels,add_self_loops=True, eps=0.9):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.eps = eps
        self.att_g = Linear(2*in_channels,1, bias=False)
        self.add_self_loops = add_self_loops
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.att_g.reset_parameters()
        # self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N+T, in_channels]
        # edge_index has shape [2, E]
        # edge_index has shape [E, weight_dim]
        edge_index , edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight,add_self_loops=self.add_self_loops,num_nodes=x.size(self.node_dim), dtype=x.dtype)
        s2s_info = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return s2s_info

    def message(self,x_i, x_j, edge_weight):
        # x_j has shape [|E|, out_channels] , The first n edges are edges of  spatial nodes
        # x_j denotes a lifted tensor, which contains the source node features of each edge, source_node(如果flow 是 source_to_target)
        # 要从从有向图的角度来解释 edge_index 有几个，就有几个x_
        
        # 对 st , ts分别定义
        alpha_i_j = self.att_g(torch.concat([x_i, x_j], axis=1)).tanh().squeeze(-1) # ( |E|, )    
        
        return x_j *( alpha_i_j * edge_weight ).view(-1,1)


class TTConv(MessagePassing):
    # convolution for relation < s -> t >
    def __init__(self, in_channels, out_channels,add_self_loops=True, eps=0.9):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.eps = eps
        self.att_g = Linear(2*in_channels,1, bias=False)
        self.add_self_loops = add_self_loops
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.att_g.reset_parameters()
        # self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N+T, in_channels]
        # edge_index has shape [2, E]
        # edge_index has shape [E, weight_dim]
        
        edge_index , edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight,add_self_loops=self.add_self_loops,num_nodes=x.size(self.node_dim), dtype=x.dtype)
        t2t_info = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return t2t_info

    def message(self,x_i, x_j, edge_weight):
        # x_j has shape [|E|, out_channels] , The first n edges are edges of  spatial nodes
        # x_j denotes a lifted tensor, which contains the source node features of each edge, source_node(如果flow 是 source_to_target)
        # 要从从有向图的角度来解释 edge_index 有几个，就有几个x_
        
        # 对 st , ts分别定义
        alpha_i_j = self.att_g(torch.concat([x_i, x_j], axis=1)).tanh().squeeze(-1) # ( |E|, )    
        
        return x_j *( alpha_i_j * edge_weight ).view(-1,1)





