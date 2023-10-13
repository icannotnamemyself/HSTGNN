import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from class_resolver.contrib.torch import activation_resolver
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import FAConv, HeteroConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.nn import HeteroConv, Linear, HANConv
from torch_timeseries.layers.weighted_hanconv import WeightedHANConv
from torch_timeseries.utils.norm import hetero_directed_norm

# from torch_timeseries.layers.graphsage import MyGraphSage


class WeightedHAN(nn.Module):
    def __init__(
        self,
        node_num,
        seq_len,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        dropout=0.0,
        norm=None,
        heads=1,
        negative_slope=0.2,
        act="relu",
        n_first=True,
        act_first=False,
        eps=0.9,
        edge_weight=True,
        conv_type='all', # homo, hetero
        **kwargs
    ):
        self.node_num = node_num
        self.seq_len = seq_len
        self.n_first = n_first

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = n_layers
        
        self.heads = heads
        self.negative_slope = negative_slope
        self.conv_type = conv_type
        self.dropout = dropout
        self.act = activation_resolver.make(act)
        self.act_first = act_first
        self.eps = eps

        self.out_channels = hidden_channels

        # assert (
        #     n_layers >= 2
        # ), "intra and inter conv layers must greater than or equals to 2 "

        
        # 定义元路径
        self.metadata = [
            ['s', 't'],
            [('s', 's2s', 's'), ('t', 't2t', 't'), ('s', 's2t', 't'),('t', 't2s', 's')]
        ]

        
        self.convs = nn.ModuleList()
        
        if n_layers > 1:
            self.convs.append(self.init_conv(self.in_channels,hidden_channels))
        
        for i in range(n_layers - 2):
            self.convs.append(self.init_conv(self.hidden_channels,hidden_channels))
        
        self.convs.append(self.init_conv(self.in_channels,out_channels))
        
        self.norms = None
        if norm is not None:
            self.norms = nn.ModuleList()
            for _ in range(n_layers - 1):
                self.norms.append(copy.deepcopy(norm))


    def init_conv(self, in_channels, out_channels, **kwargs):
        han = WeightedHANConv(in_channels, out_channels,self.metadata,heads=self.heads, negative_slope=self.negative_slope,dropout=self.dropout)
        return han
    
    def forward(self, x, edge_index, edge_attr):
        # x: B * (N+T) * C
        # edge_index: B,2,2*(N*T)
        # edge_attr: B*E or B * (N * T )

        for i in range(self.num_layers):
            xs = list()
            for bi in range(x.shape[0]):
                x_dict = {
                    "s": x[bi][: self.node_num, :],
                    "t": x[bi][self.node_num :, :],
                }
                edge_index_bi = edge_index[bi]
                edge_weight_bi = edge_attr[bi]
                # TODO: edge may be empty, please ensure no empty edges here
                assert ((edge_index_bi[0] < self.node_num) 
                    & (edge_index_bi[1] < self.node_num)).any() == True
                
                nn_index = (edge_index_bi[0] < self.node_num)  & (edge_index_bi[1] < self.node_num)
                tt_index = (edge_index_bi[0] >= self.node_num) & (edge_index_bi[1] >= self.node_num)
                nt_index = (edge_index_bi[0] < self.node_num)  & (edge_index_bi[1] >= self.node_num)
                tn_index = (edge_index_bi[0] >= self.node_num) & (edge_index_bi[1] < self.node_num)
                
                edge_nn = edge_index_bi[:,nn_index,]
                edge_nn_weight = edge_weight_bi[nn_index,]
                
                edge_tt = edge_index_bi[:,tt_index,] 
                edge_tt_weight = edge_weight_bi[tt_index,]
                
                edge_nt = edge_index_bi[:,nt_index,]
                edge_nt_weight = edge_weight_bi[nt_index,]

                edge_tn = edge_index_bi[:,tn_index,]
                edge_tn_weight = edge_weight_bi[tn_index,]
   
                # convert edge index to edge index dict
                    
                edge_tt = edge_tt - self.node_num 
                edge_nt[1, :] = edge_nt[1, :] - self.node_num
                edge_tn[0, :] = edge_tn[0, :] - self.node_num
                
                
                edge_index_dict = {}
                if self.conv_type == 'all':
                    edge_index_dict = {
                        ("s", "s2s", "s"): edge_nn,
                        ("t", "t2t", "t"): edge_tt,
                        ("s", "s2t", "t"): edge_nt,
                        ("t", "t2s", "s"): edge_tn,
                    }
                    
                    edge_weight_dict = {
                        ("s", "s2s", "s"): edge_nn_weight,
                        ("t", "t2t", "t"): edge_tt_weight,
                        ("s", "s2t", "t"): edge_nt_weight,
                        ("t", "t2s", "s"): edge_tn_weight,
                    }
                elif self.conv_type == 'homo':
                    
                    edge_index_dict = {
                        ("s", "s2s", "s"): edge_nn,
                        ("t", "t2t", "t"): edge_tt,
                    }
                    edge_weight_dict = {
                        ("s", "s2s", "s"): edge_nn_weight,
                        ("t", "t2t", "t"): edge_tt_weight,
                    }

                elif self.conv_type == 'hetero':
                    edge_index_dict = {
                        ("s", "s2t", "t"): edge_nt,
                        ("t", "t2s", "s"): edge_tn,
                    }
                    edge_weight_dict = {
                        ("s", "s2t", "t"): edge_nt_weight,
                        ("t", "t2s", "s"): edge_tn_weight,
                    }
                else:
                    raise NotImplementedError("conv_type must be 'all', 'homo' or 'heter'")
                out_dict,out_att = self.convs[i](x_dict, edge_index_dict,edge_weight_dict)
                xi = torch.concat([out_dict["s"], out_dict["t"]], dim=0)
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

