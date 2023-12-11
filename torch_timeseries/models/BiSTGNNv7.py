from typing import List, Tuple
import torch
from torch import nn
from torch_timeseries.layers.heterostgcn5 import HeteroFASTGCN as HeteroFASTGCN5
from torch_timeseries.layers.heterostgcn6 import HeteroFASTGCN as HeteroFASTGCN6
from torch_timeseries.layers.han import HAN
from torch_timeseries.layers.tcn_output2 import TCNOuputLayer as TCNOuputLayer2
from torch_timeseries.layers.tcn_output3 import TCNOuputLayer as TCNOuputLayer3
from torch_timeseries.layers.tcn_output4 import TCNOuputLayer as TCNOuputLayer4
from torch_timeseries.layers.tcn_output import TCNOuputLayer
from torch_timeseries.layers.weighted_han import WeightedHAN
from torch_timeseries.layers.weighted_han_update import WeightedHAN as WeightedHANUpdate


from builtins import print
import torch
from torch._C import get_num_interop_threads
import torch.nn as nn
import pyro

class BiSTGNNv7(nn.Module):
    def __init__(
        self,
        seq_len,
        num_nodes,
        temporal_embed_dim,
        graph_build_type="adaptive", # "predefined_adaptive"
        graph_conv_type="weighted_han",
        output_layer_type='tcn6',
        heads=1,
        negative_slope=0.2,
        gcn_layers=2,
        tn_layers=2,
        rebuild_time=True,
        rebuild_space=True,
        node_static_embed_dim=32,
        latent_dim=32,
        tcn_layers=5,
        dilated_factor=2,
        tcn_channel=16,
        out_seq_len=1,
        dropout=0.0,
        predefined_adj=None,
        act='elu',
        self_loop_eps=0.5,
        without_tn_module=False,
    ):
        super(BiSTGNNv7, self).__init__()

        self.num_nodes = num_nodes
        self.static_embed_dim = node_static_embed_dim
        self.latent_dim = latent_dim
        self.tn_layers = tn_layers
        self.heads = heads
        self.negative_slope = negative_slope
        self.out_seq_len = out_seq_len
        self.self_loop_eps = self_loop_eps
        self.without_tn_module = without_tn_module

        self.spatial_encoder = SpatialEncoder(
            seq_len,
            num_nodes,
            static_embed_dim=node_static_embed_dim,
            latent_dim=latent_dim,
        )
        self.temporal_encoder = TemporalEncoder(
            seq_len,
            num_nodes,
            temporal_embed_dim,
            static_embed_dim=node_static_embed_dim,
            latent_dim=latent_dim,
        )
        

        self.rebuild_time = rebuild_time
        self.rebuild_space = rebuild_space

        # Rebuild Module
        if rebuild_space:
            self.feature_rebuild = nn.Sequential(
                nn.Linear(latent_dim, seq_len),
                torch.nn.ELU(),
                nn.Linear(seq_len, seq_len),
            )

        if rebuild_time:
            self.time_rebuild = nn.GRU(latent_dim, num_nodes, batch_first=True)
        
        self.tn_modules = nn.ModuleList()
        
        if not self.without_tn_module:
            for i in range(self.tn_layers):
                self.tn_modules.append(
                    TNModule(
                        num_nodes,
                        seq_len,
                        latent_dim,
                        gcn_layers,
                        graph_build_type=graph_build_type,
                        graph_conv_type=graph_conv_type,
                        heads=self.heads,
                        negative_slope=self.negative_slope,
                        dropout=dropout,
                        act=act,
                        predefined_adj=predefined_adj,
                        self_loop_eps=self.self_loop_eps
                    )
                )

        out_channels = 1
        if rebuild_space:
            out_channels = out_channels + 1
        if rebuild_time:
            out_channels = out_channels + 1
        if output_layer_type == 'tcn6':
            self.output_layer = TCNOuputLayer4(
                input_seq_len=seq_len,
                num_nodes=self.num_nodes,
                out_seq_len=self.out_seq_len,
                tcn_layers=tcn_layers,
                dilated_factor=dilated_factor,
                in_channel=out_channels,
                tcn_channel=tcn_channel,
            )
        else:
            raise NotImplementedError(f"output layer type {output_layer_type} not implemented")



    def forward(self, x, x_enc_mark):
        """
        in :  (B, N, T)
        out:  (B, N, latent_dim)
        """

        Xs = self.spatial_encoder(x)
        Xt = self.temporal_encoder(x.transpose(1, 2), x_enc_mark)
        X = torch.concat([Xs, Xt], dim=1)  # (B, N+T, latent_dim)
        
        if not self.without_tn_module:
            for i in range(self.tn_layers):
                X = self.tn_modules[i](X)  # (B, N+T, latent_dim)

        # rebuild module
        Xs = X[:, : self.num_nodes, :]  # (B, T, D)
        Xt = X[:, self.num_nodes :, :]  # (B, T, D)
        outputs = list()
        if self.rebuild_space:
            n_output = self.feature_rebuild(Xs)  # (B, N, T)
            outputs.append(n_output.unsqueeze(1))
        if self.rebuild_time:
            t_output,_ = self.time_rebuild(Xt)  # (B, T, N)
            outputs.append(t_output.unsqueeze(1).transpose(2, 3))
        outputs.append(x.unsqueeze(1))  # （B, 1/2/3, N, T)

        # output module
        X = torch.cat(outputs, dim=1)  # （B, 1/2/3, N, T)
        X = self.output_layer(X)  # (B, N+T, 1)

        return X


class TNModule(nn.Module):
    def __init__(
        self, num_nodes, seq_len, latent_dim, gcn_layers, graph_build_type="adaptive",
        graph_conv_type='fastgcn5',heads=1,negative_slope=0.2,dropout=0.0,act='elu',self_loop_eps=0.5,
        predefined_adj=None
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act=  act
        self.self_loop_eps = self_loop_eps

        if graph_build_type == "adaptive":
            self.graph_constructor = STGraphConstructor(self_loop_eps=self.self_loop_eps, dim_feats=latent_dim, dim_h=latent_dim, dim_z=latent_dim)
        elif graph_build_type == "predefined_adaptive":
            assert predefined_adj is not None, "predefined_NN_adj must be provided"
            self.graph_constructor = STGraphConstructor(predefined_adj=predefined_adj,self_loop_eps=self.self_loop_eps, dim_feats=latent_dim, dim_h=latent_dim, dim_z=latent_dim)
        elif graph_build_type == "fully_connected":
            self.graph_constructor = STGraphConstructor(predefined_adj=None, adaptive=False,self_loop_eps=self.self_loop_eps, dim_feats=latent_dim, dim_h=latent_dim, dim_z=latent_dim)
            print("graph_build_type is fully_connected")

        if graph_conv_type == 'fastgcn5':
            self.graph_conv = HeteroFASTGCN5(
                num_nodes, seq_len, latent_dim, latent_dim, gcn_layers, dropout=0,act='elu'
            )
        elif graph_conv_type == 'fastgcn6':
            self.graph_conv = HeteroFASTGCN6(
                num_nodes, seq_len, latent_dim, latent_dim, gcn_layers, dropout=0,act='elu'
            )
        elif graph_conv_type == 'han':
            self.graph_conv = HAN(
                num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act
            )
        elif graph_conv_type == 'han_homo':
            self.graph_conv = HAN(
                num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type='homo'
            )
        elif graph_conv_type == 'han_hetero':
            self.graph_conv = HAN(
                num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type='hetero'
            )
        elif graph_conv_type == 'weighted_han':
            self.graph_conv = WeightedHAN(
                num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type='all'
            )
        elif graph_conv_type == 'weighted_han_update':
            print("graph_conv_type","weighted_han_update")
            self.graph_conv = WeightedHANUpdate(
                num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type='all'
            )
        elif graph_conv_type == 'weighted_han_homo':
            self.graph_conv = WeightedHAN(
                num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type='homo'
            )
        elif graph_conv_type == 'weighted_han_hetero':
            self.graph_conv = WeightedHAN(
                num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type='hetero'
            )
        else:
            raise NotImplementedError("Unknown graph_conv")
        self.graph_embedding = GraphEmbeeding(latent_dim=latent_dim)

    def forward(self, X):
        """
        Args:
            X :  (B, N+T, latent_dim)

        Returns:
            X: (B, N+T, latent_dim)
        """
        Xs = X[:, : self.num_nodes, :]
        Xt = X[:, self.num_nodes :, :]

        batch_adj, batch_indices, batch_values = self.graph_constructor(
            Xs, Xt
        )  # spatial and temporal adjecent matrix
        X = self.graph_conv(X, batch_indices, batch_values)
        X = self.graph_embedding(batch_adj, X)

        return X


class SpatialEncoder(nn.Module):
    def __init__(self, seq_len, num_nodes, static_embed_dim=32, latent_dim=256):
        super(SpatialEncoder, self).__init__()

        self.num_nodes = num_nodes
        self.static_embed_dim = static_embed_dim
        self.latent_dim = latent_dim

        self.static_node_embedding = nn.Embedding(num_nodes, static_embed_dim)
        input_dim = seq_len + static_embed_dim
        self.spatial_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            torch.nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        """
        in :  (B, N, T)
        out:  (B, N, latent_dim)
        """

        # 获取输入张量x的形状
        B, N, T = x.size()
        assert N == self.num_nodes
        # set dtype to LongType
        static_embeding = self.static_node_embedding.weight.expand(B, N, -1)

        # 将static_embedding和x沿最后一个维度连接，得到(B, N, T+static_embed_dim)的新张量
        x = torch.cat((x, static_embeding), dim=2)

        # 对combined_x进行进一步处理和计算
        spatial_encode  = self.spatial_projection(x)

        return spatial_encode


class TemporalEncoder(nn.Module):
    def __init__(
        self, seq_len, num_nodes, temporal_embed_dim, static_embed_dim=32, latent_dim=64
    ):
        super(TemporalEncoder, self).__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.static_embed_dim = static_embed_dim
        self.static_node_embedding = nn.Embedding(seq_len, static_embed_dim)
        self.temporal_embed_dim = temporal_embed_dim
        if self.temporal_embed_dim  > 0:
            input_dim = num_nodes + static_embed_dim + temporal_embed_dim
        else:
            input_dim = num_nodes + static_embed_dim
        self.temporal_projection = nn.GRU(
            input_dim,latent_dim, batch_first=True
        )

    def forward(self, x, x_enc_mark):
        """
        x :  (B, T, N)
        x_enc_mark : (B, T, D_t)


        out:  (B, T, latent_dim)
        """
        # 获取输入张量x的形状
        B, T, N = x.size()
        assert T == self.seq_len
        static_embeding = self.static_node_embedding.weight.expand(B, T, -1)
        if self.temporal_embed_dim  > 0:
            # 将static_embedding和x沿最后一个维度连接，得到(B, T, N+D_t +static_embed_dim)的新张量
            x = torch.concat((x, x_enc_mark, static_embeding), dim=2)
        else:
            x = torch.concat((x, static_embeding), dim=2)

        # 对combined_x进行进一步处理和计算
        temporal_encode , _= self.temporal_projection(x)
        return temporal_encode


class STGraphConstructor(nn.Module):
    def __init__(self, dim_feats, dim_h, dim_z, predefined_adj=None, adaptive=True,self_loop_eps=0.5):
        super(STGraphConstructor, self).__init__()
        self.predefined_adj =predefined_adj
        self.adaptive =adaptive
        self.self_loop_eps = self_loop_eps
        self.dim_feats = dim_feats
        self.dim_h = dim_h
        self.dim_z = dim_z
        
        
        # self.vae = VGAE(dim_feats=self.dim_feats , dim_h=self.dim_h,dim_z=self.dim_z,activation=torch.nn.ReLU(), gae=True)
        
    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes,  temporal_nodes], dim=1)
        
        

        if not self.adaptive:
            batch_adj = torch.ones(B, N+T, N+T).to(node_embs.device)
        else:
            batch_adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs)
            if self.predefined_adj is not None:
                batch_adj = batch_adj + self.predefined_adj # enhance adj with prior
            batch_adj = torch.tanh( torch.relu(batch_adj)) 
            
            # adjs = []
            # for i, adj in enumerate(batch_adj):
            #     adj_logits = self.vae(adj , node_embs[i])
            #     adj_new = sample_adj(adj_logits)   
            #     adjs.append(adj_new)
            # batch_adj = torch.stack(adjs) # (B, N+T, N+T)
        # extract to pyg format
        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = batch_adj[bi].nonzero().t()
            edge_weights = batch_adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return batch_adj, batch_indices, batch_values


class GraphEmbeeding(nn.Module):
    def __init__(self, latent_dim):
        super(GraphEmbeeding, self).__init__()

    def forward(self, adj, X) -> Tuple[torch.Tensor, torch.Tensor]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        return X






#     """ GAE/VGAE as edge prediction model """
# class VGAE(nn.Module):
#     def __init__(self, dim_feats, dim_h, dim_z, activation, gae=False):
#         super(VGAE, self).__init__()
#         self.gae = gae
#         self.layers = nn.ModuleList()
#         self.layers.append(GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False))
#         self.layers.append(GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False))

#     def forward(self, adj, features):
#         # GCN encoder
#         hidden = self.layers[0](adj, features)
#         self.mean = self.layers[1](adj, hidden)
#         if self.gae:
#             # GAE (no sampling at bottleneck)
#             Z = self.mean
#         else:
#             # VGAE
#             self.logstd = self.layers[2](adj, hidden)
#             gaussian_noise = torch.randn_like(self.mean)
#             sampled_Z = gaussian_noise*torch.exp(self.logstd) + self.mean
#             Z = sampled_Z
#         # inner product decoder
#         adj_logits = Z @ Z.T
#         return adj_logits

# class GCNLayer(nn.Module):
#     """ one layer of GCN """
#     def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
#         super(GCNLayer, self).__init__()
#         self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
#         self.activation = activation
#         if bias:
#             self.b = nn.Parameter(torch.FloatTensor(output_dim))
#         else:
#             self.b = None
#         if dropout:
#             self.dropout = nn.Dropout(p=dropout)
#         else:
#             self.dropout = 0
#         self.init_params()

#     def init_params(self):
#         """ Initialize weights with xavier uniform and biases with all zeros """
#         for param in self.parameters():
#             if len(param.size()) == 2:
#                 nn.init.xavier_uniform_(param)
#             else:
#                 nn.init.constant_(param, 0.0)

#     def forward(self, adj, h):
#         if self.dropout:
#             h = self.dropout(h)
#         x = h @ self.W
#         x = adj @ x
#         if self.b is not None:
#             x = x + self.b
#         if self.activation:
#             x = self.activation(x)
#         return x


# def sample_adj(adj_logits):
#     """ sample an adj from the predicted edge probabilities of ep_net """
#     relu = torch.nn.ReLU()
#     adj_logits = relu(adj_logits)
#     adj_logits_ = (adj_logits / torch.max(adj_logits))
#     # sampling
#     adj_sampled = adj_logits_ * pyro.distributions.RelaxedBernoulliStraightThrough(temperature=0.2, probs=adj_logits_).rsample()
#     # making adj_sampled symmetric
#     return adj_sampled



