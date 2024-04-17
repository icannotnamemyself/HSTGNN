
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from class_resolver.contrib.torch import activation_resolver
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear



from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor

class MAGNN(nn.Module):
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
        conv = MAGNNConv(in_channels, out_channels,self.metadata,heads=self.heads, negative_slope=self.negative_slope,dropout=self.dropout)
        return conv


    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_layers):
            xs = list()
            for bi in range(x.shape[0]):
                x_dict = {
                    "s": x[bi][: self.node_num, :],
                    "t": x[bi][self.node_num :, :],
                }
                edge_index_bi = edge_index[bi]
                edge_weight_bi = edge_attr[bi]
                
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
                elif self.conv_type == 'ss':
                    edge_index_dict = {
                        ("s", "s2s", "s"): edge_nn,
                    }
                    edge_weight_dict = {
                        ("s", "s2s", "s"): edge_nn_weight,
                    }
                elif self.conv_type == 'tt':
                    
                    edge_index_dict = {
                        ("t", "t2t", "t"): edge_tt,
                    }
                    edge_weight_dict = {
                        ("t", "t2t", "t"): edge_tt_weight,
                    }
                elif self.conv_type == 'st':
                    edge_index_dict = {
                        ("s", "s2t", "t"): edge_nt,
                    }
                    edge_weight_dict = {
                        ("s", "s2t", "t"): edge_nt_weight,
                    }
                elif self.conv_type == 'ts':
                    edge_index_dict = {
                        ("t", "t2s", "s"): edge_tn,
                    }
                    edge_weight_dict = {
                        ("t", "t2s", "s"): edge_tn_weight,
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
                out_dict = self.convs[i](x_dict, edge_index_dict)
                # out_dict,out_att = self.convs[i](x_dict, edge_index_dict,edge_weight_dict)
                
                
                if self.conv_type == 'ss':
                    xi = torch.concat([out_dict["s"], x[bi][self.node_num: , :]], dim=0)
                elif self.conv_type == 'tt':
                    xi = torch.concat([x[bi][:self.node_num , :] , out_dict["t"]], dim=0)
                else:
                    xi = torch.concat([out_dict["s"], out_dict["t"]], dim=0)
                xs.append(xi)
                
            x = torch.stack(xs)
            if i == self.num_layers - 1:
                break

            # if self.norms is not None:
            #     x = self.norms[i](x)

            x = self.act(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
        return x




class GraphAttention(nn.Module):
    """Attention layer for encoded metapaths"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        weighted, _ = self.attention(queries, keys, values)
        return weighted


class MetapathEncoder(nn.Module):
    """Inter-metapath aggregator"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, metapath):
        x = torch.mean(metapath, dim=1)
        return self.fc(x)





class MAGNNConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        edge_dim=1,
        normalize_emb=False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.edge_dim = edge_dim
        self.dropout = dropout
        
        
        self.intra_metapath_encoder = nn.ModuleDict()
        self.intra_metapath_attention = nn.ModuleDict()
        self.inter_metapath_encoder = nn.ModuleDict()
        

        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)
            self.inter_metapath_encoder[node_type] = MetapathEncoder(out_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.intra_metapath_encoder[edge_type] = MetapathEncoder(out_channels, out_channels)
            self.intra_metapath_attention[edge_type] = GraphAttention(out_channels, 1, dropout)
            
    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
    ) -> Union[Dict[NodeType, OptTensor], Tuple[Dict[NodeType, OptTensor],
                                                Dict[NodeType, OptTensor]]]:
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}
        
        out_list = []

        # Iterate over node types:
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            out = self.propagate(edge_index, x=(x_src, x_dst), size=None, edge_type=edge_type)

            out_dict[dst_type].append(out)
            out_list.append(out)
        # iterate over node types:
        for node_type, outs in out_dict.items():
            out_dict[node_type] = self.inter_metapath_encoder[node_type](torch.stack(outs, dim=1))
        return out_dict

    def message(self, x_j: Tensor,x_i: Tensor, edge_type) -> Tensor:
        metapath = torch.cat([
            x_j,
            x_i
            ], dim=1
        )
        aggregated_metapath = F.tanh(self.intra_metapath_encoder[edge_type](metapath))
        # aggregated_metapath = self.intra_metapath_attention[edge_type](aggregated_metapath)
        return aggregated_metapath

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')
        

    # def update(self, aggr_out, x,edge_type):
    #     # aggr_out has shape [N, out_channels]
    #     # x has shape [N, in_channels]
    #     (x_src, x_dst) = x
    #     aggr_out = self.intra_metapath_attention[edge_type](torch.cat((aggr_out.unsqueeze(1), x_dst),dim=1))
    #     return aggr_out
