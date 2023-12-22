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
from torch_timeseries.utils.norm import hetero_directed_norm

# from torch_timeseries.layers.graphsage import MyGraphSage

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax



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
                # assert ((edge_index_bi[0] < self.node_num) 
                    # & (edge_index_bi[1] < self.node_num)).any() == True
                
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
                out_dict,out_att = self.convs[i](x_dict, edge_index_dict,edge_weight_dict)
                
                
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

            # if self.act_first:
            #     x = self.act(x)
            # if self.norms is not None:
            #     x = self.norms[i](x)
            # if not self.act_first:
            #     x = self.act(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
        return x






def group(
    xs: List[Tensor],
    q: nn.Parameter,
    k_lin: nn.Module,
) -> Tuple[OptTensor, OptTensor]:

    if len(xs) == 0:
        return None, None
    else:
        num_edge_types = len(xs)
        out = torch.stack(xs)
        if out.numel() == 0:
            return out.view(0, out.size(-1)), None
        attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out, attn


class WeightedHANConv(MessagePassing):
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
        
        self.q = nn.ParameterDict({node_type: nn.Parameter(torch.Tensor(1, out_channels))for node_type in metadata[0]  })
        self.k_lin = nn.ModuleDict({node_type:nn.Linear(out_channels, out_channels) for node_type in metadata[0]  })

        
        self.agg_lin_dict =  nn.ModuleDict()
        self.edge_att_dict =  nn.ModuleDict()
        self.normalize_emb = normalize_emb

        self.proj = nn.ModuleDict()
        self.edge_proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.edge_proj[edge_type] = nn.Linear(self.edge_dim, out_channels)
            self.agg_lin_dict[edge_type] =  nn.Linear(in_channels +out_channels, out_channels)
            self.edge_att_dict[edge_type] =  nn.Linear(dim +dim+dim, out_channels)
            
        self.act = torch.nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        for node_type in self.metadata[0]:
            glorot(self.q[node_type])
            self.k_lin[node_type].reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        # return_semantic_attention_weights: bool = False,
        edge_weight_dict:Dict[EdgeType, Adj],
        
    ) -> Union[Dict[NodeType, OptTensor], Tuple[Dict[NodeType, OptTensor],
                                                Dict[NodeType, OptTensor]]]:
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Iterate over node types:
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():
            edge_weight = edge_weight_dict[edge_type]
            edge_weight = edge_weight.view(edge_weight.shape[0], self.edge_dim)
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            edge_weight = self.edge_proj[edge_type](edge_weight).view(-1, H, D)
            # import pdb;pdb.set_trace()
            lin_src = self.lin_src[edge_type]
            lin_dst = self.lin_dst[edge_type]
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            # propagate_type: (x_dst: PairTensor, alpha: PairTensor)
            out = self.propagate(edge_index, x=(x_src, x_dst),
                                 alpha=(alpha_src, alpha_dst),edge_weight=edge_weight, size=None, edge_type=edge_type)

            # out = F.relu(out)
            out = F.tanh(out)
            out_dict[dst_type].append(out)

        # iterate over node types:
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            out, attn = group(outs, self.q[node_type], self.k_lin[node_type])
            out_dict[node_type] = out
            semantic_attn_dict[node_type] = attn

        return out_dict, semantic_attn_dict

    def message(self, x_j: Tensor,x_i, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int],edge_weight:Optional[Tensor],edge_type) -> Tensor:

        alpha = alpha_j + alpha_i
        att = self.edge_att_dict[edge_type](torch.concat((x_j,x_i,edge_weight), dim=-1))
        if edge_weight is not None:
            alpha = alpha 
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1) * att
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')
    def update(self, aggr_out, x,edge_type):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        (x_src, x_dst) = x
        x_src = x_src.squeeze()
        x_dst = x_dst.squeeze()
        aggr_out = self.act(self.agg_lin_dict[edge_type](torch.cat((aggr_out, x_dst),dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out
