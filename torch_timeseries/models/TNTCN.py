import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from class_resolver.contrib.torch import activation_resolver
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import FAConv


class TNTCN(nn.Module):
    def __init__(self, n_nodes, input_seq_len,remain_prob=1.0,graph_build_type='weighted_random_clip',output_module='tcn',gcn_type='fagcn',gcn_eps=0.1,casting_dim=32,gcn_channel=32, gc_layers=2, edge_mode=1, aggr_mode='add', dropout=0.3,act='elu',tcn_channel=16,pred_horizon=3,multi_pred=False,no_time=False,no_space=False, tcn_layers=3,one_node_forecast=False,dilated_factor=2,
                without_gc=False):
        super().__init__()
        self.act = activation_resolver.make(act)

        self.graph_build_type = graph_build_type
        self.without_gc = without_gc
        if not without_gc:
            self.tn_module = TNModule(n_nodes, input_seq_len,remain_prob,gcn_type,gcn_eps,casting_dim,gcn_channel, gc_layers, edge_mode, aggr_mode, dropout,act,tcn_channel,pred_horizon,multi_pred,no_time ,no_space, dilated_factor, tcn_layers,one_node_forecast,graph_build_type=graph_build_type)

        out_channels = 1
        if not without_gc:
            out_channels = 3
            if no_time or no_space:
                out_channels = 2
        
        out_seq_len = pred_horizon if multi_pred else 1
        
        if output_module == 'tcn':
            self.output_module = TCNOuputLayer(
                input_seq_len=input_seq_len,
                out_seq_len=out_seq_len,
                tcn_layers=tcn_layers,
                dilated_factor=dilated_factor,
                in_channel=out_channels,
                tcn_channel=tcn_channel,
                act=act)
        elif output_module == 'mlp':
            self.output_module = MLPOuputLayer(
                input_seq_len=input_seq_len,
                out_seq_len=out_seq_len,
                in_channel=out_channels,
                hidden_dim=tcn_channel,
                act=act
            )
        
        # self.channel_layer = nn.Conv2d(out_channels, tcn_channel, (1, 1))
        # self.end_layer = nn.Conv2d(tcn_channel, 1, (1, 1))

        self.has_node_layer = False
        if one_node_forecast:
            self.node_layer = nn.Linear(n_nodes, 1)
            self.has_node_layer = True

    def forward(self, x, edge_mask=None):
        # x.shape: (batch, n_nodes, seq_len)
        if not self.without_gc:
            output = self.tn_module(x, edge_mask)
        else:
            output = x
            # output = self.gc_replace(x)
            output = output.unsqueeze(1) 
            
        # output: (B, 2 or 3, N, T)
        output = self.output_module(output) # (B, out_len, N)

        if self.has_node_layer:
            output = self.node_layer(output)
        output = output.squeeze(1)  # (B, N) or (B, out_len, N)
        return output

class TCNOuputLayer(nn.Module):
    def __init__(self,input_seq_len,out_seq_len,tcn_layers,dilated_factor,in_channel,tcn_channel,act='elu') -> None:
        super().__init__()
        self.channel_layer = nn.Conv2d(in_channel, tcn_channel, (1, 1))
        self.tcn =MyTCN(
                    input_seq_len, tcn_channel, tcn_channel,
                    out_seq_len=out_seq_len, num_layers=tcn_layers,
                    dilated_factor=dilated_factor
                )
        self.end_layer = nn.Conv2d(tcn_channel, 1, (1, 1))
        self.act = activation_resolver.make(act)
        
    def forward(self, x):
        output = self.channel_layer(x)
        output = self.act(self.tcn(output))  # (B, C, N, out_len)
        output = self.end_layer(output).squeeze(1).transpose(1, 2)  # (B, out_len, N)
        return output
        
        

class MLPOuputLayer(nn.Module):
    def __init__(self,input_seq_len,out_seq_len,in_channel,hidden_dim,act='elu') -> None:
        super().__init__()
        self.mlp1 = nn.Linear(input_seq_len*in_channel, hidden_dim) 
        self.act = activation_resolver.make(act)
        self.mlp2 = nn.Linear(hidden_dim, out_seq_len) 
        
    def forward(self, x):
        x = x.reshape(x.size(0),x.size(2),-1) # (B, C, N, T) -> (B, N, TxC)
        
        output = self.act(self.mlp1(x)) # (B, N, H)
        output = self.mlp2(output) # (B, N,out_seq_len)
        
        output = output.transpose(1, 2) # (B,out_seq_len, N)
        return output
        
class TNMLP(nn.Module):
    def __init__(self, n_nodes, input_seq_len,remain_prob,gcn_type,gcn_eps,casting_dim,gcn_channel, gc_layers, edge_mode, aggr_mode, dropout,act,tcn_channel,pred_horizon,multi_pred,no_time ,no_space, dilated_factor, tcn_layers,one_node_forecast
                 ,without_gc,
                 num_classes) -> None:
        super().__init__()

        self.without_gc = without_gc
        if not self.without_gc:
            self.tn_module = TNModule(n_nodes, input_seq_len,remain_prob,gcn_type,gcn_eps,casting_dim,gcn_channel, gc_layers, edge_mode, aggr_mode, dropout,act,tcn_channel,pred_horizon,multi_pred,no_time ,no_space, dilated_factor, tcn_layers,one_node_forecast)
        self.act = activation_resolver.make(act)
        
        mlp_input_dim = 3 * n_nodes * input_seq_len
        if self.without_gc:
            mlp_input_dim = n_nodes * input_seq_len
        hidden_dim = int(math.sqrt(mlp_input_dim * num_classes))

        self.classifi_layer = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_mask=None):
        batch, _, _ = x.shape
        if not self.without_gc:
            x = self.tn_module(x, edge_mask)  # (B, 3, N, T)
        x = self.classifi_layer(x.reshape(batch, -1))  # (B, n_classes)
        # output = torch.max(output, dim=1)  # （B, n_classes)
        return x


class TNModule(nn.Module):
    def __init__(self, n_nodes, input_seq_len,remain_prob,gcn_type,gcn_eps,casting_dim,gcn_channel, gc_layers, edge_mode, aggr_mode, dropout,act,tcn_channel,pred_horizon,multi_pred,no_time ,no_space, dilated_factor, tcn_layers,one_node_forecast,graph_build_type='weighted_random_clip' ):
        # graph_build_type: weighted_random_clip full_connected
        
        super().__init__()
        
        

        self.n_nodes = n_nodes
        self.seq_len = input_seq_len
        
        self.graph_build_type = graph_build_type

        def get_edge_index():
            tmp = torch.zeros((self.n_nodes + self.seq_len, self.n_nodes + self.seq_len)) # (NxT , NxT)
            tmp[:self.n_nodes, self.n_nodes:] = 1
            tmp[self.n_nodes:, :self.n_nodes] = 1
            return torch.nonzero(tmp).T
        self.edge_index = get_edge_index()
        self.remain_prob = remain_prob

        self.act = activation_resolver.make(act)

        # Casting Module
        self.feature_cast = nn.Sequential(
            nn.Linear(self.seq_len, casting_dim),
            self.act,
            nn.Linear(casting_dim, casting_dim)
        )
        self.time_cast = nn.GRU(self.n_nodes, casting_dim)

        # Graph Convolution
        self.gcn_type = gcn_type
        if gcn_type == 'sage':
            self.gcn = MyGraphSage(
                casting_dim, gcn_channel, gc_layers, 
                act=act, edge_mode=edge_mode, normalize_emb=False, 
                aggr=aggr_mode, dropout=dropout, eps=gcn_eps
            )
        elif gcn_type == 'fagcn':
            self.gcn = MyFAGCN(
                casting_dim, gcn_channel, gc_layers,
                act=act, eps=gcn_eps
            )

        # Rebuild Module
        if not no_time:
            self.time_rebuild = nn.GRU(gcn_channel, self.n_nodes, batch_first=True)
        if not no_space:
            self.feature_rebuild = nn.Sequential(
                nn.Linear(gcn_channel, self.seq_len),
                self.act,
                nn.Linear(self.seq_len, self.seq_len)
            )
        # self.channel_layer = nn.Conv2d(1 if without_gc else 3, gcn_channel, (1, 1))
    
    def build_graph_for_pyg(self, x, remain_mask=None):
        # input: B N T
        # output: 
            # edge_index: B , 2, 2*(N+T)
            # edge_attr: B , 2*(N+T)
        batch_size = x.shape[0]
        batch_indices = self.edge_index.expand(batch_size, -1, -1).to(x.device)
        # indices = torch.nonzero(adj).T  # (2, n_edges)
        batch_values = torch.cat((
            x.reshape(batch_size, -1), x.transpose(1, 2).reshape(batch_size, -1)
        ), dim=-1)  # (batch_size, n_edges)  # (B, 2(NxT))

        if self.remain_prob < 1.0:
            if self.training and remain_mask is not None:
                batch_indices = batch_indices[:, :, remain_mask]
                batch_values = batch_values[:, remain_mask]
        
        elif self.graph_build_type == 'weighted_random_clip':
            sampler = torch.tanh(torch.abs(batch_values))  # (B, n_edges)
            sampler = torch.bernoulli(sampler)
            sampler = sampler == 1
            res = list()
            for bi in range(batch_size):
                ei = batch_indices[bi]# (第i行 第j列) 有边
                sample_indx = sampler[bi]
                res.append(torch.stack((ei[0, sample_indx], ei[1, sample_indx])))
            batch_indices = res

        elif self.graph_build_type == 'full_connected':
            batch_indices = batch_indices
            
        else:
            raise NotImplementedError("Graph constructor not implemented!!!")
        
        
        return batch_indices, batch_values

    def forward(self, x, edge_mask=None):
        """
        x.shape: (batch, n_nodes, seq_len) -> output.shape: (batch, 3, n_nodes, seq_len)
        """
        feature_nodes = self.feature_cast(x)  # (B, N, H)
        time_nodes, _ = self.time_cast(x.transpose(1, 2))  # (B, T, H)

        edge_index, edge_attr = self.build_graph_for_pyg(x, edge_mask)
        node_embs = torch.cat((feature_nodes, time_nodes), dim=1)  # (B, N + T, H)

        output = self.gcn(node_embs , edge_index ,edge_attr=edge_attr) # (B, N + T, H)
        n_output = output[:, :self.n_nodes, :]  # (B, N, C_out)
        t_output = output[:, self.n_nodes:, :]  # (B, T, C_out)
        
        outputs = list()
        if hasattr(self, 'feature_rebuild'):
            n_output = self.feature_rebuild(n_output)  # (B, N, T)
            outputs.append(n_output.unsqueeze(1))
        if hasattr(self, 'time_rebuild'):
            t_output, _ = self.time_rebuild(t_output)  # (B, T, N)
            outputs.append(t_output.unsqueeze(1).transpose(2, 3))
        outputs.append(x.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B, 2 or 3, N, T)
        # output = torch.cat((
        #     n_output.unsqueeze(1), t_output.unsqueeze(1).transpose(2, 3), x.unsqueeze(1)
        # ), dim=1)  # (B, 3, N, T)
        # # output = self.channel_layer(output)  # (B, C, N, T)
        # return output


class MyGCN(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, n_layers, 
        out_channels=None, dropout=0, norm=None,
        act='relu', act_first=False, **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = n_layers

        self.dropout = dropout
        self.act = activation_resolver.make(act)
        self.act_first = act_first

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
        return GraphConvolution(in_channels, out_channels, batch_adj=kwargs.get('batch_adj', False))

    def forward(self, x, adj):
        origin = x
        for i in range(self.num_layers):
            x = self.convs[i](x, adj)
            if i == self.num_layers - 1:
                break
            
            if self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if not self.act_first:
                x = self.act(x)
            
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = origin + x
        return x


class MyGraphSage(nn.Module):
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

    def forward(self, x, edge_attr, edge_index):
        if len(edge_attr.shape) == 2:
            edge_attr = edge_attr.unsqueeze(-1)  # (B, E) -> (B, E, 1)
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


class MyFAGCN(MyGraphSage):
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
        if len(edge_attr.shape) == 2:
            edge_attr = edge_attr.unsqueeze(-1)  # (B, E) -> (B, E, 1)
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


class MyMixGCN(MyGCN):
    def __init__(self, in_channels, hidden_channels, n_layers, out_channels=None, dropout=0, norm=None, act='relu', act_first=False, **kwargs):
        super().__init__(in_channels, hidden_channels, n_layers, out_channels, dropout, norm, act, act_first, **kwargs)

    def init_conv(self, in_channels, out_channels, **kwargs):
        return MixProp(in_channels, out_channels, dropout=self.dropout, **kwargs)


class MyTCN(nn.Module):
    def __init__(
        self, seq_len, in_channels, hidden_channels, out_seq_len=1, out_channels=None, num_layers=2, 
        dilated_factor=2, dropout=0, act='relu'
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dropout = dropout
        self.act = activation_resolver.make(act)
        self.num_layers = num_layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.filter_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=dilated_factor))
        self.gate_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=dilated_factor))
        for _ in range(num_layers - 2):
            self.filter_convs.append(DilatedInception(hidden_channels, hidden_channels, dilation_factor=dilated_factor))
            self.gate_convs.append(DilatedInception(hidden_channels, hidden_channels, dilation_factor=dilated_factor))
        
        if out_channels is not None:
            self.filter_convs.append(DilatedInception(hidden_channels, out_channels, dilation_factor=dilated_factor))
            self.gate_convs.append(DilatedInception(hidden_channels, out_channels, dilation_factor=dilated_factor))
        else:
            self.filter_convs.append(DilatedInception(hidden_channels, hidden_channels, dilation_factor=dilated_factor))
            self.gate_convs.append(DilatedInception(hidden_channels, hidden_channels, dilation_factor=dilated_factor))
    
        max_kernal_size = 7
        receptive_field = num_layers * dilated_factor * (max_kernal_size - 1) + 1
        self.min_input_len = receptive_field + out_seq_len - 1

        end_kernel_size = 1
        if seq_len > self.min_input_len:
            end_kernel_size = seq_len - self.min_input_len + 1
        self.end_conv = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=out_channels if out_channels is not None else hidden_channels, 
            kernel_size=(1, end_kernel_size), bias=True
        )


    def forward(self, x):
        """
        x.shape: (B, Cin, N, T)

        output.shape: (B, Cout, N, out_seq_len)
        """
        batch, _, n_nodes, seq_len = x.shape
        assert seq_len == self.seq_len

        if seq_len < self.min_input_len:
            x = nn.functional.pad(x, (self.min_input_len - seq_len, 0, 0, 0))

        origin = x
        for i in range(self.num_layers):
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            x = x + origin[:, :, :, -x.shape[3]:]
        
        x = self.end_conv(x)
        
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_channel, out_channel, add_self_loop=True, normalize=True, bias=True, batch_adj=False):
        super(GraphConvolution, self).__init__()

        self.add_self_loop = add_self_loop
        self.normalize = normalize
        self.batch_adj = batch_adj
        self.lin = Linear(in_channel, out_channel, bias=False, weight_initializer='glorot')

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channel))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, inputs, adj):
        """
        Args:
            inputs: (batch, N, in_channels)
            # adj: (batch, N, N)
            adj: (N, N)
        
        Return:
            output: (batch, N, out_channels)
        """
        _, n_nodes, _ = inputs.shape

        # add self-loop
        if self.add_self_loop:
            adj = adj + torch.eye(n_nodes).to(adj.device)
        
        # normalization
        if self.normalize:
            adj = adj / adj.sum(-1).unsqueeze(-1)
        
        inputs = self.lin(inputs)  # (batch, N, out_channel)

        # message passing
        if self.batch_adj:
            output = torch.einsum('bnc, bvn->bvc', (inputs, adj))  # (B, N, out_channel)
        else:
            output = torch.einsum('bnc, vn->bvc', (inputs, adj))

        if self.bias is not None:
            output = output + self.bias
        
        return output


# class FAConv(nn.Module):
#     def __init__(self, in_channel, out_channel, add_self_loop=True, normalize=True, bias=True):
#         super().__init__()
#         self.add_self_loop = add_self_loop
#         self.normalize = normalize
#         self.lin = Linear(in_channel, out_channel, bias=False, weight_initializer='glorot')
#         self.gate = Linear(2 * in_channel, 1)

#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channel))
#         else:
#             self.register_parameter('bias', None)
    
#     def forward(self, inputs, adj):
#         """
#         Args:
#             inputs: (batch, N, in_channels)
#             adj: (N, N)
#         Return:
#             output: (batch, N, out_channels)
#         """
#         batch, n_nodes, Cin = inputs.shape

#         inputs = inputs.permute(1, 2, 0)  # (N, C, B)
#         ex_inputs = inputs.expand(n_nodes, n_nodes, Cin, batch)  # (N, N, C, B)
#         ex_inputs = torch.cat((ex_inputs.transpose(0, 1), ex_inputs), dim=-2)  # (N, N, 2C, B)
#         ex_inputs = ex_inputs.permute(3, 0, 1, 2)  # (B, N, N, 2C)
#         adj_gates = torch.tanh(self.gate(ex_inputs).squeeze(-1))  # (B, N, N)

#         # add self-loop
#         if self.add_self_loop:
#             adj = adj + torch.eye(n_nodes).to(adj.device)
        
#         # normalization
#         if self.normalize:
#             adj = adj / adj.sum(-1).unsqueeze(-1)
        
#         adj = adj_gates * adj
        
#         inputs = self.lin(inputs)  # (batch, N, out_channel)

#         # message passing
#         output = torch.einsum('bnc, bvn->bvc', (inputs, adj))  # (B, N, out_channel)
#         # output = torch.einsum('bnc, vn->bvc', (inputs, adj))

#         if self.bias is not None:
#             output = output + self.bias
        
#         return output


class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep=2, alpha=0.05, add_self_loop=True, normalize=True, dropout=0, oper=None, **kwargs):
        super(MixProp, self).__init__()
        # self.end_conv = nn.Conv2d((gdep + 1) * c_in, c_out, (1, 1))
        self.end_layer = nn.Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.alpha = alpha
        self.add_self_loop = add_self_loop
        self.normalize = normalize
        self.dropout = dropout

        # self.oper = 'bnc,bvn->bvc' if batch_adj else 'bnc,vn->bvc'
        self.oper = 'bnc,vn->bvc' if oper is None else oper
    
    def nconv(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x

    def forward(self, x, adj):
        n_nodes = adj.shape[-1]

        # add self-loop
        if self.add_self_loop:
            adj = adj + torch.eye(n_nodes).to(adj.device)
        
        # normalization
        if self.normalize:
            adj = adj / adj.sum(-1).unsqueeze(-1)
        
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = (1 - self.alpha) * F.dropout(torch.einsum(self.oper, (h, adj)), p=self.dropout, training=self.training)
            h += self.alpha * x
            out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.end_layer(ho)
        return ho


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


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        # input.shape: (B, C, N, T)
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))

        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


class DilatedInception1d(nn.Module):
    kernel_set = [2, 3, 6, 7]
    max_kernel_size = max(kernel_set)

    def __init__(self, cin, cout, dilation_factor=2, dropout=0.3, act='relu'):
        super(DilatedInception1d, self).__init__()
        self.tconv = nn.ModuleList()
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv1d(cin, cout, kern, dilation=dilation_factor))
        self.dropout = nn.Dropout(dropout)
        self.act = activation_resolver.make(act)

    def forward(self, input):
        # input.shape: (B, C, T)
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(-1):]
        x = torch.cat(x,dim=1)
        x = self.act(x)
        x = self.dropout(x)
        return x


class DilatedInception2d(nn.Module):
    kernel_set = [2, 3, 6, 7]
    max_kernel_size = max(kernel_set)

    def __init__(self, cin, cout, dilation_factor=2, dropout=0.3, act='relu'):
        super(DilatedInception2d, self).__init__()
        self.tconv = nn.ModuleList()
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=dilation_factor))
        self.dropout = nn.Dropout(dropout)
        self.act = activation_resolver.make(act)

    def forward(self, input):
        # input.shape: (B, C, T)
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(-1):]
        x = torch.cat(x,dim=1)
        x = self.act(x)
        x = self.dropout(x)
        return x
