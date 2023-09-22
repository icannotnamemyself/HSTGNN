from typing import List, Tuple
import torch
from torch import nn
from torch_timeseries.layers.heterostgcn5 import HeteroFASTGCN as HeteroFASTGCN5
from torch_timeseries.layers.heterostgcn6 import HeteroFASTGCN as HeteroFASTGCN6
from torch_timeseries.layers.han import HAN
from torch_timeseries.layers.tcn_output import TCNOuputLayer


class BiSTGNNv2(nn.Module):
    def __init__(
        self,
        seq_len,
        num_nodes,
        temporal_embed_dim,
        graph_build_type="adaptive",
        graph_conv_type="han",
        heads=1,
        negative_slope=0.2,
        gcn_layers=2,
        tn_layers=2,
        rebuild_time=True,
        rebuild_space=True,
        node_static_embed_dim=32,
        latent_dim=32,
        tcn_layers=3,
        dilated_factor=2,
        tcn_channel=16,
        dropout=0.0,
        act='elu'
    ):
        super(BiSTGNNv2, self).__init__()

        self.num_nodes = num_nodes
        self.static_embed_dim = node_static_embed_dim
        self.latent_dim = latent_dim
        self.tn_layers = tn_layers
        self.heads = heads
        self.negative_slope = negative_slope

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
                )
            )

        out_channels = 1
        if rebuild_space:
            out_channels = out_channels + 1
        if rebuild_time:
            out_channels = out_channels + 1

        self.output_layer = TCNOuputLayer(
            input_seq_len=seq_len,
            out_seq_len=1,
            tcn_layers=tcn_layers,
            dilated_factor=dilated_factor,
            in_channel=out_channels,
            tcn_channel=tcn_channel,
        )

    def forward(self, x, x_enc_mark):
        """
        in :  (B, N, T)
        out:  (B, N, latent_dim)
        """

        Xs = self.spatial_encoder(x)
        Xt = self.temporal_encoder(x.transpose(1, 2), x_enc_mark)
        X = torch.concat([Xs, Xt], dim=1)  # (B, N+T, latent_dim)
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
        graph_conv_type='fastgcn5',heads=1,negative_slope=0.2,dropout=0.0,act='elu'
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act=  act

        if graph_build_type == "adaptive":
            self.graph_constructor = STGraphConstructor()
            
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
        input_dim = num_nodes + static_embed_dim + temporal_embed_dim
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

        # 将static_embedding和x沿最后一个维度连接，得到(B, T, N+D_t +static_embed_dim)的新张量
        x = torch.concat((x, x_enc_mark, static_embeding), dim=2)

        # 对combined_x进行进一步处理和计算
        temporal_encode , _= self.temporal_projection(x)
        return temporal_encode


class STGraphConstructor(nn.Module):
    def __init__(self):
        super(STGraphConstructor, self).__init__()

    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()

        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        adj = torch.relu(torch.einsum("bnf, bmf -> bnm", node_embs, node_embs))
        # add self loop

        # Create a unit matrix and expand its dimensions to match the dimensions of x
        eye = torch.eye(N+T,N+T).to(adj.device)  # Ensure the unit matrix is on the same device as x
        eye_expanded = eye.unsqueeze(0).repeat(B, 1, 1)  # Expand the unit matrix to shape (B, N, N)
        adj = torch.tanh(adj + eye_expanded)

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            adj.nonzero().t()
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values


class GraphEmbeeding(nn.Module):
    def __init__(self, latent_dim):
        super(GraphEmbeeding, self).__init__()

    def forward(self, adj, X) -> Tuple[torch.Tensor, torch.Tensor]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        return X
