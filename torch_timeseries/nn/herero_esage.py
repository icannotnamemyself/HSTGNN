import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch.nn.init import xavier_uniform_, zeros_

import torch.nn as nn
import torch.nn.functional as F

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError


class GNNStack(torch.nn.Module):
    def __init__(self, 
                node_input_dim, edge_input_dim,
                node_dim, edge_dim, dropout, activation,
                gnn_layer_num,num_nodes
                ):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.gnn_layer_num =gnn_layer_num
        self.num_nodes = num_nodes
        self.edge_update_mlps = self.build_edge_update_mlps(node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)
        
        # convs
        self.convs = self.build_convs(node_input_dim, edge_input_dim,
                                    node_dim, edge_dim, activation)
    def build_convs(self, node_input_dim, edge_input_dim,
                     node_dim, edge_dim,  activation):
        convs = nn.ModuleList()
        conv = self.build_conv_model(node_input_dim,node_dim,
                                    edge_input_dim , activation)
        convs.append(conv)
        for l in range(1, self.gnn_layer_num):
            conv = self.build_conv_model(node_dim, node_dim,
                                    edge_dim, activation)
            convs.append(conv)
        return convs
    def build_conv_model(self, node_in_dim, node_out_dim, edge_dim, activation):
            return HeteroSTEGraphSage(self.num_nodes,node_in_dim,node_out_dim,edge_dim,activation)
    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims, dropout, activation):
        if 0 in hidden_dims:
            return get_activation('none')
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layer = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            get_activation(activation),
                            nn.Dropout(dropout),
                            )
                layers.append(layer)
                input_dim = hidden_dim
            layer = nn.Linear(input_dim, output_dim)
            layers.append(layer)
            return nn.Sequential(*layers)
        
    def forward(self,x, edge_attr, edge_index):
        # x : (N+M x M) node emebedding
        concat_x = []
        for l,conv in enumerate(self.convs):
            edge_index_dict, edge_weight_dict = self.edge_index_extraction(edge_index, edge_attr)
            x = conv(x, edge_weight_dict, edge_index_dict)
            concat_x.append(x)
            
            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
        x = torch.cat(concat_x, 1)
        return x
    
    def update_edge_attr(self, x, edge_attr, edge_index, mlp):

        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat((x_i,x_j,edge_attr),dim=-1))
        return edge_attr

    def build_edge_update_mlps(self, node_dim, edge_input_dim, edge_dim, gnn_layer_num, activation):
        edge_update_mlps = nn.ModuleList()
        edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_input_dim,edge_dim),
                get_activation(activation),
                )
        edge_update_mlps.append(edge_update_mlp)
        for l in range(1,gnn_layer_num):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_dim,edge_dim),
                get_activation(activation),
                )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps


    def edge_index_extraction(self, edge_index_bi, edge_weight_bi):
        nt_index = (edge_index_bi[0] < self.num_nodes)  & (edge_index_bi[1] >= self.num_nodes)
        tn_index = (edge_index_bi[0] >= self.num_nodes) & (edge_index_bi[1] < self.num_nodes)
        
        edge_nt = edge_index_bi[:,nt_index,]
        edge_nt_weight = edge_weight_bi[nt_index,]

        edge_tn = edge_index_bi[:,tn_index,]
        edge_tn_weight = edge_weight_bi[tn_index,]

        # convert edge index to edge index dict
            
        edge_nt[1, :] = edge_nt[1, :] - self.num_nodes
        edge_tn[0, :] = edge_tn[0, :] - self.num_nodes
        
        edge_index_dict = {
            ("s", "s2t", "t"): edge_nt,
            ("t", "t2s", "s"): edge_tn,
        }
        edge_weight_dict = {
            ("s", "s2t", "t"): edge_nt_weight,
            ("t", "t2s", "s"): edge_tn_weight,
        }
        return edge_index_dict, edge_weight_dict


        # post node updat
class HeteroSTEGraphSage(MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, num_nodes, in_channels, out_channels,
                 edge_channels, activation):
        super(HeteroSTEGraphSage, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.num_nodes = num_nodes
        self.message_lin_dict = nn.ModuleDict({
            's':  nn.Linear(in_channels+edge_channels, out_channels),
            't':  nn.Linear(in_channels+edge_channels, out_channels),
        })
        

        self.agg_lin_dict = nn.ModuleDict({
            's':  nn.Linear(in_channels+out_channels, out_channels),
            't':  nn.Linear(in_channels+out_channels, out_channels),
        })
        
        
        self.update_activation = get_activation(activation)
        self.message_activation = get_activation(activation)

    def forward(self, x, edge_attr_dict, edge_index_dict):
        xs = x[:self.num_nodes]
        xt = x[self.num_nodes:]
        num_nodes = x.size(0)
        edge_type = ("s", "s2t", "t")
        xt =  self.propagate(edge_index_dict[edge_type], x=x, edge_attr=edge_attr_dict[edge_type], size=(num_nodes, num_nodes),node_type='t')
        edge_type = ("t", "t2s", "s")
        xs = self.propagate(edge_index_dict[edge_type], x=x, edge_attr=edge_attr_dict[edge_type], size=(num_nodes, num_nodes),node_type='s')
        x = torch.concat([xs[:self.num_nodes], xt[self.num_nodes:]], dim=0)
        return x
        

    def message(self, x_i, x_j, edge_attr, edge_index, size, node_type):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]
        m_j = torch.cat((x_j, edge_attr),dim=-1)
        m_j = self.message_activation(self.message_lin_dict[node_type](m_j))
        return m_j

    def update(self, aggr_out, x, node_type):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        aggr_out = self.update_activation(self.agg_lin_dict[node_type](torch.cat((aggr_out, x),dim=-1)))
        aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out












