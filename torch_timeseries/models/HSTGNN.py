from typing import List, Tuple
import torch
from torch import nn
from torch_timeseries.layers.heterostgcn5 import HeteroFASTGCN as HeteroFASTGCN5
from torch_timeseries.layers.heterostgcn6 import HeteroFASTGCN as HeteroFASTGCN6
from torch_timeseries.layers.han import HAN
from torch_timeseries.layers.tcn_output2 import TCNOuputLayer as TCNOuputLayer2
from torch_timeseries.layers.tcn_output3 import TCNOuputLayer as TCNOuputLayer3
from torch_timeseries.layers.tcn_output4 import TCNOuputLayer as TCNOuputLayer4
from torch_timeseries.layers.tcn_output8 import TCNOuputLayer as TCNOuputLayer8
from torch_timeseries.layers.tcn_output import TCNOuputLayer
from torch_timeseries.layers.weighted_han import WeightedHAN
from torch_timeseries.layers.weighted_han_update import WeightedHAN as WeightedHANUpdate
from torch_timeseries.layers.graphsage import MyGraphSage, MyFAGCN


class HSTGNN(nn.Module):
    def __init__(
        self,
        seq_len,
        num_nodes,
        temporal_embed_dim,
        graph_build_type="adaptive", # "predefined_adaptive"
        graph_conv_type="weighted_han",
        output_layer_type='tcn8',
        heads=1,
        negative_slope=0.2,
        gcn_layers=2,
        rebuild_time=True,
        rebuild_space=True,
        node_static_embed_dim=16,
        latent_dim=16,
        tcn_layers=5,
        dilated_factor=2,
        tcn_channel=16,
        out_seq_len=1,
        dropout=0.0,
        predefined_adj=None,
        act='elu',
        self_loop_eps=0.1,
        without_tn_module=False,
        d0=2,
        kernel_set=[2,3,6,7]
    ):
        super(HSTGNN, self).__init__()

        self.num_nodes = num_nodes
        self.static_embed_dim = node_static_embed_dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.out_seq_len = out_seq_len
        self.self_loop_eps = self_loop_eps
        self.without_tn_module = without_tn_module
        self.kernel_set = kernel_set

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
            for i in range(1):
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
            if out_seq_len == 1:
                self.output_layer = TCNOuputLayer(
                    input_seq_len=seq_len,
                    out_seq_len=self.out_seq_len,
                    tcn_layers=tcn_layers,
                    dilated_factor=dilated_factor,
                    in_channel=out_channels,
                    tcn_channel=tcn_channel,
                )
            else:
                self.output_layer = TCNOuputLayer4(
                    input_seq_len=seq_len,
                    num_nodes=self.num_nodes,
                    out_seq_len=self.out_seq_len,
                    tcn_layers=tcn_layers,
                    dilated_factor=dilated_factor,
                    in_channel=out_channels,
                    tcn_channel=tcn_channel,
                )
        elif output_layer_type == 'tcn7':
            self.output_layer = TCNOuputLayer4(
                input_seq_len=seq_len,
                num_nodes=self.num_nodes,
                out_seq_len=self.out_seq_len,
                tcn_layers=tcn_layers,
                dilated_factor=dilated_factor,
                in_channel=out_channels,
                tcn_channel=tcn_channel,
            )
        elif output_layer_type == 'tcn8':
            self.output_layer = TCNOuputLayer8(
                input_seq_len=seq_len,
                num_nodes=self.num_nodes,
                out_seq_len=self.out_seq_len,
                tcn_layers=tcn_layers,
                dilated_factor=dilated_factor,
                in_channel=out_channels,
                tcn_channel=tcn_channel,
                d0=d0,
                kernel_set=kernel_set,
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
            for i in range(1):
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
            self.graph_constructor = STGraphConstructor(self_loop_eps=self.self_loop_eps)
        elif graph_build_type == "predefined_adaptive":
            assert predefined_adj is not None, "predefined_NN_adj must be provided"
            self.graph_constructor = STGraphConstructor(predefined_adj=predefined_adj,self_loop_eps=self.self_loop_eps)
        elif graph_build_type == "fully_connected":
            self.graph_constructor = STGraphConstructor(predefined_adj=None, adaptive=False,self_loop_eps=self.self_loop_eps)
            print("graph_build_type is fully_connected")

        # if graph_conv_type == 'gcn':
        #     pass
        if graph_conv_type == 'graphsage':
            self.graph_conv = MyGraphSage(latent_dim, latent_dim, gcn_layers,act='elu')
        elif graph_conv_type == 'fagcn':
            self.graph_conv = MyFAGCN(latent_dim, latent_dim, gcn_layers,act='elu')
        elif graph_conv_type == 'han':
            self.graph_conv = HAN(
                num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act
            )
        elif graph_conv_type == 'weighted_han':
            self.graph_conv = WeightedHAN(
                num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type='all'
            )
        elif graph_conv_type == 'weighted_han_update':
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
    def __init__(self, predefined_adj=None, adaptive=True,self_loop_eps=0.5):
        super(STGraphConstructor, self).__init__()
        self.predefined_adj =predefined_adj
        self.adaptive =adaptive
        self.self_loop_eps = self_loop_eps
        
    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        

        if not self.adaptive:
            adj = torch.ones(B, N+T, N+T).to(node_embs.device)
        else:
            
            # predefined adjcent matrix        
            if self.predefined_adj is not None:
                # avoid inplace operation 
                adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) + self.predefined_adj
            else:
                adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) 
            
            
            adj = torch.tanh( torch.relu(adj))
            eye = torch.eye(N+T,N+T).to(adj.device)  # Ensure the unit matrix is on the same device as x
            eye_expanded = eye.unsqueeze(0).repeat(B, 1, 1)  # Expand the unit matrix to shape (B, N, N)
            adj =adj + eye_expanded * self.self_loop_eps
    
            # # predefined adjcent matrix        
            # if self.predefined_NN_adj is not None:
            #     # avoid inplace operation 
            #     new_adj = adj.clone()
            #     new_adj[:, :N, :N] = new_adj[:, :N, :N] * self.normalized_predefined_adj.repeat(B, 1, 1)
            #     adj = new_adj

            # eye_expanded = eye.unsqueeze(0).repeat(B, 1, 1)  # Expand the unit matrix to shape (B, N, N)
            # adj = torch.tanh(adj + eye_expanded)
        
        

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values

