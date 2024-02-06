from typing import List, Tuple
import pyro
import torch
from torch import nn
from torch_timeseries.layers.heterostgcn5 import HeteroFASTGCN as HeteroFASTGCN5
from torch_timeseries.layers.heterostgcn6 import HeteroFASTGCN as HeteroFASTGCN6
from torch_timeseries.layers.han import HAN
from torch_timeseries.layers.hgt import HGT
from torch_timeseries.layers.tcn_output2 import TCNOuputLayer as TCNOuputLayer2
from torch_timeseries.layers.tcn_output3 import TCNOuputLayer as TCNOuputLayer3
from torch_timeseries.layers.tcn_output4 import TCNOuputLayer as TCNOuputLayer4
from torch_timeseries.layers.tcn_output8 import TCNOuputLayer as TCNOuputLayer8
from torch_timeseries.layers.tcn_output9_norm import TCNOuputLayer as TCNOuputLayer9
from torch_timeseries.layers.tcn_output import TCNOuputLayer
from torch_timeseries.layers.nlinear_output import NlinearOuputLayer
from torch_timeseries.layers.weighted_han import WeightedHAN
from torch_timeseries.layers.weighted_han_update import WeightedHAN as WeightedHANUpdate
from torch_timeseries.layers.weighted_han_update2 import WeightedHAN as WeightedHANUpdate2
from torch_timeseries.layers.weighted_han_update3 import WeightedHAN as WeightedHANUpdate3
from torch_timeseries.layers.weighted_han_update4 import WeightedHAN as WeightedHANUpdate4
from torch_timeseries.layers.weighted_han_update5 import WeightedHAN as WeightedHANUpdate5
from torch_timeseries.layers.weighted_han_update6 import WeightedHAN as WeightedHANUpdate6
from torch_timeseries.layers.weighted_han_update7 import WeightedHAN as WeightedHANUpdate7
from torch_timeseries.layers.weighted_han_update8 import WeightedHAN as WeightedHANUpdate8
from torch_timeseries.layers.weighted_han_update9 import WeightedHAN as WeightedHANUpdate9
from torch_timeseries.layers.graphsage import MyGraphSage, MyFAGCN


class HSTGNN(nn.Module):
    def __init__(
        self,
        seq_len,
        num_nodes,
        temporal_embed_dim,
        graph_build_type="attsim_direc_tt_mask2", # "predefined_adaptive"
        graph_conv_type="weighted_han_update9",
        output_layer_type='tcn8',
        heads=1,
        negative_slope=0.2,
        gcn_layers=2,
        rebuild_time=True,
        rebuild_space=True,
        node_static_embed_dim=16,
        latent_dim=32,
        tcn_layers=5,
        dilated_factor=2,
        tcn_channel=16,
        out_seq_len=1,
        dropout=0.0,
        predefined_adj=None,
        act='elu',
        self_loop_eps=0.1,
        without_tn_module=False,
        without_gcn=False,
        d0=2,
        kernel_set=[2,3,6,7],
        normalization=True,
        conv_type='all'
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
        self.normalization =normalization
        self.conv_type = conv_type
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
                        conv_type=conv_type,
                        heads=self.heads,
                        negative_slope=self.negative_slope,
                        dropout=dropout,
                        act=act,
                        predefined_adj=predefined_adj,
                        self_loop_eps=self.self_loop_eps,
                        without_gcn=without_gcn
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
                act=act,
                d0=d0,
                kernel_set=kernel_set,
            )
        elif output_layer_type == 'tcn9':
            self.output_layer = TCNOuputLayer9(
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
        if self.normalization:
            seq_last = x[:,:,-1:].detach()
            x = x - seq_last
        # import pdb;pdb.set_trace()
        # residual = x

        Xs = self.spatial_encoder(x)
        Xt = self.temporal_encoder(x.transpose(1, 2), x_enc_mark)
        X = torch.concat([Xs, Xt], dim=1)  # (B, N+T, latent_dim)
        
        if not self.without_tn_module:
            for i in range(1):
                X = self.tn_modules[i](X)  # (B, N+T, latent_dim)

        # rebuild module
        Xs = X[:, : self.num_nodes, :]  # (B, N, D)
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
        X = self.output_layer(X)  # (B, O, N)
        
        
        if self.normalization:
            X = (X.transpose(1,2) + seq_last).transpose(1,2)

        return X


class TNModule(nn.Module):
    def __init__(
        self, num_nodes, seq_len, latent_dim, gcn_layers, graph_build_type="adaptive",
        graph_conv_type='fastgcn5',conv_type='all',heads=1,negative_slope=0.2,dropout=0.0,act='elu',self_loop_eps=0.5,without_gcn=False,
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
        self.without_gcn =without_gcn
        self.conv_type = conv_type
        if graph_build_type == "adaptive":
            self.graph_constructor = STGraphConstructor(self_loop_eps=self.self_loop_eps)
        elif graph_build_type == "predefined_adaptive":
            assert predefined_adj is not None, "predefined_NN_adj must be provided"
            self.graph_constructor = STGraphConstructor(predefined_adj=predefined_adj,self_loop_eps=self.self_loop_eps)
        elif graph_build_type == "fully_connected":
            self.graph_constructor = STGraphConstructor(predefined_adj=None, adaptive=False,self_loop_eps=self.self_loop_eps)
            print("graph_build_type is fully_connected")
        elif graph_build_type == "mlp":
            self.graph_constructor = MLPConstructor(predefined_adj=predefined_adj,latent_dim=latent_dim,self_loop_eps=self.self_loop_eps)
        elif graph_build_type == "mlp2":
            self.graph_constructor = MLPConstructor2(predefined_adj=predefined_adj,latent_dim=latent_dim,self_loop_eps=self.self_loop_eps)
            print("graph_build_type dis mlp2")

        elif graph_build_type == "mlpsim":
            self.graph_constructor = MLPSimConstructor(predefined_adj=predefined_adj,latent_dim=latent_dim,tt_mask=False)
            print("graph_build_type dis mlpsim")
        elif graph_build_type == "mlpsim_tt_mask":
            self.graph_constructor = MLPSimConstructor(predefined_adj=predefined_adj,latent_dim=latent_dim,tt_mask=True)
            print("graph_build_type dis mlpsim_tt_mask")
        elif graph_build_type == "mlpsim_direc_tt_mask":
            self.graph_constructor = MLPSimDirectConstructor(predefined_adj=predefined_adj,latent_dim=latent_dim,tt_mask=True)
            print("graph_build_type dis mlpsim_direc_tt_mask")
        elif graph_build_type == "mlpsim_direc_tt_mask_norm":
            self.graph_constructor = MLPSimDirectNormConstructor(predefined_adj=predefined_adj,latent_dim=latent_dim,tt_mask=True)
            print("graph_build_type dis MLPSimDirectNormConstructor")

        elif graph_build_type == "mlpsim_direc_tt_mask_sample":
            self.graph_constructor = MLPSimSampleDirectConstructor(predefined_adj=predefined_adj,latent_dim=latent_dim,tt_mask=True)
            print("graph_build_type dis mlpsim_tt_mask1")

        elif graph_build_type == "attsim_direc_tt_mask1":
            self.graph_constructor = AttentionSimDirectConstructor(predefined_adj=predefined_adj,latent_dim=latent_dim,tt_mask=True, N=num_nodes, O=seq_len)
            print("graph_build_type dis mlpsim_direc_tt_mask_sample")
        elif graph_build_type == "attsim_direc_tt_mask2":
            self.graph_constructor = AttentionSimDirectConstructor2(predefined_adj=predefined_adj,latent_dim=latent_dim,tt_mask=True, N=num_nodes, O=seq_len)
            print("graph_build_type dis attsim_direc_tt_mask2")
        elif graph_build_type == "attsim_direc_tt_mask3":
            self.graph_constructor = AttentionSimDirectConstructor3(predefined_adj=predefined_adj,latent_dim=latent_dim,tt_mask=True, N=num_nodes, O=seq_len)
            print("graph_build_type dis attsim_direc_tt_mask3")
        elif graph_build_type == "attsim_direc_tt_mask3_no_predefined":
            self.graph_constructor = AttentionSimDirectConstructor3(predefined_adj=None,latent_dim=latent_dim,tt_mask=True, N=num_nodes, O=seq_len)
            print("graph_build_type dis attsim_direc_tt_mask3_no_predefined")
        else:
            raise NotImplementedError(f"grpah build type {graph_build_type} not implemented")



        # if graph_conv_type == 'gcn':
        #     pass
        if not self.without_gcn:
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
                    conv_type=self.conv_type
                )
            elif graph_conv_type == 'weighted_han_update2':
                self.graph_conv = WeightedHANUpdate2(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                    conv_type='all'
                )
            elif graph_conv_type == 'weighted_han_update3':
                self.graph_conv = WeightedHANUpdate3(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                    conv_type=self.conv_type
                )
            elif graph_conv_type == 'weighted_han_update4':
                self.graph_conv = WeightedHANUpdate4(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                    conv_type=self.conv_type
                )
            elif graph_conv_type == 'weighted_han_update5':
                self.graph_conv = WeightedHANUpdate5(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                    conv_type=self.conv_type
                )
            elif graph_conv_type == 'weighted_han_update6':
                self.graph_conv = WeightedHANUpdate6(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                    conv_type=self.conv_type
                )
            elif graph_conv_type == 'weighted_han_update7':
                self.graph_conv = WeightedHANUpdate7(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                    conv_type=self.conv_type
                )
            elif graph_conv_type == 'weighted_han_update8':
                self.graph_conv = WeightedHANUpdate8(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                    conv_type=self.conv_type
                )
            elif graph_conv_type == 'weighted_han_update9':
                self.graph_conv = WeightedHANUpdate9(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                    conv_type=self.conv_type
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
                    conv_type=self.conv_type
                )
            elif graph_conv_type == 'hgt':
                self.graph_conv = HGT(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type=self.conv_type
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
        if not self.without_gcn:
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



class MLPConstructor(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,self_loop_eps=0.5):
        super(MLPConstructor, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        
        
        self.relation_mlp_model = nn.ModuleDict({
            'ss': nn.Linear(latent_dim+latent_dim, 1),
            'tt': nn.Linear(latent_dim+latent_dim, 1),
            'st': nn.Linear(latent_dim+latent_dim, 1),
            'ts': nn.Linear(latent_dim+latent_dim, 1),
        })
        self.mask_mlp_model = nn.ModuleDict({
            'ss': nn.Linear(latent_dim+latent_dim, 1),
            'tt': nn.Linear(latent_dim+latent_dim, 1),
            'st': nn.Linear(latent_dim+latent_dim, 1),
            'ts': nn.Linear(latent_dim+latent_dim, 1),
        })
    
    def build_sub_graph(self, x1, x2, relation_name):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj = self.relation_mlp_model[relation_name](x_combined)  # [B, N*O, 1]
        adj = adj.reshape(B, N1, N2)  # [B, N, O]
        
        mask = self.mask_mlp_model[relation_name](x_combined) # [B, N*O, 1]
        mask = torch.relu(mask)
        mask = mask.reshape(B, N1, N2)  # [B, N, O]

        
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj, mask

    def build_graph(self, spatial_nodes, temporal_nodes):
        
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        ss_adj, ss_mask = self.build_sub_graph(spatial_nodes, spatial_nodes,'ss' )
        tt_adj, tt_mask = self.build_sub_graph(temporal_nodes, temporal_nodes,'tt' )
        st_adj, st_mask  = self.build_sub_graph(spatial_nodes, temporal_nodes,'st' )
        ts_adj, ts_mask = self.build_sub_graph(temporal_nodes, spatial_nodes,'ts' )
        upper_row = torch.cat([ss_adj, st_adj], dim=-1)
        lower_row = torch.cat([ts_adj, tt_adj], dim=-1)
        # 然后垂直拼接这两行以形成最终的大矩阵
        adj_matrix = torch.cat([upper_row, lower_row], dim=-2)



        mask_upper_row = torch.cat([ss_mask, st_mask], dim=-1)
        mask_lower_row = torch.cat([ts_mask, tt_mask], dim=-1)

        # 然后垂直拼接这两行以形成最终的大矩阵
        mask = torch.cat([mask_upper_row, mask_lower_row], dim=-2)

        return adj_matrix, mask



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        _adj, mask = self.build_graph(spatial_nodes, temporal_nodes)
        # adj = _adj * mask
        adj = torch.tanh(_adj * mask)

        # predefined adjcent matrix        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            adj = adj + self.predefined_adj
        

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values
















class MLPConstructor2(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,self_loop_eps=0.5):
        super(MLPConstructor2, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        
        
        self.relation_mlp_model = nn.ModuleDict({
            'ss': nn.Linear(latent_dim+latent_dim, 1),
            'tt': nn.Linear(latent_dim+latent_dim, 1),
            'st': nn.Linear(latent_dim+latent_dim, 1),
            'ts': nn.Linear(latent_dim+latent_dim, 1),
        })

    def build_sub_graph(self, x1, x2, relation_name):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj = self.relation_mlp_model[relation_name](x_combined)  # [B, N*O, 1]
        adj = adj.reshape(B, N1, N2)  # [B, N, O]

        
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj

    def build_graph(self, spatial_nodes, temporal_nodes):
        
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        ss_adj  = self.build_sub_graph(spatial_nodes, spatial_nodes,'ss' )
        tt_adj  = self.build_sub_graph(temporal_nodes, temporal_nodes,'tt' )
        st_adj  = self.build_sub_graph(spatial_nodes, temporal_nodes,'st' )
        ts_adj  = self.build_sub_graph(temporal_nodes, spatial_nodes,'ts' )
        upper_row = torch.cat([ss_adj, st_adj], dim=-1)
        lower_row = torch.cat([ts_adj, tt_adj], dim=-1)
        # 然后垂直拼接这两行以形成最终的大矩阵
        adj_matrix = torch.cat([upper_row, lower_row], dim=-2)

        return adj_matrix



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        _adj = self.build_graph(spatial_nodes, temporal_nodes)
        adj = torch.tanh(torch.relu(_adj))

        # predefined adjcent matrix        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            adj = adj + self.predefined_adj
        

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values











class MLPSimConstructor(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,tt_mask=True):
        super(MLPSimConstructor, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        self.tt_mask = tt_mask
        self.relation_mlp_model = nn.ModuleDict({
            'st': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
            'ts': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
        })

    
    def build_sub_graph(self, x1, x2, relation_name):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj = self.relation_mlp_model[relation_name](x_combined)  # [B, N*O, 1]
        adj = adj.reshape(B, N1, N2)  # [B, N, O]
        
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 


    def build_graph(self, spatial_nodes, temporal_nodes):
        
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) 

        st_adj  = self.build_sub_graph(spatial_nodes, temporal_nodes,'st' )
        ts_adj = self.build_sub_graph(temporal_nodes, spatial_nodes,'ts' )
        adj[:, :N, N:] = st_adj
        adj[:, N:, :N] = ts_adj
        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            adj = adj + self.predefined_adj
        
        if self.tt_mask:
            adj[:, N:, N:] = torch.triu(adj[:, N:, N:])


        
        return adj



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        _adj = self.build_graph(spatial_nodes, temporal_nodes)
        # adj = _adj * mask
        adj = torch.tanh(torch.relu(_adj))

        # predefined adjcent matrix        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            adj = adj + self.predefined_adj
        

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values








class MLPSimDirectNormConstructor(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,tt_mask=True, alpha=3):
        super(MLPSimDirectNormConstructor, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        self.tt_mask = tt_mask
        self.relation_mlp_model = nn.ModuleDict({
            'st': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
            'ts': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
        })
        
        
                
        self.relation_mlp_model = nn.ModuleDict({
            'ss': nn.Linear(latent_dim+latent_dim, 1),
            'tt': nn.Linear(latent_dim+latent_dim, 1),
            'st': nn.Linear(latent_dim+latent_dim, 1),
            'ts': nn.Linear(latent_dim+latent_dim, 1),
        })

        self.alpha = alpha
        
        
        self.ss_x1_lin = nn.Linear(latent_dim, latent_dim, bias=False)
        self.ss_x2_lin = nn.Linear(latent_dim, latent_dim, bias=False)
    
    def build_sub_graph(self, x1, x2, relation_name):
        
        if relation_name =='ss':
            return self.build_ss_graph(x1, x2)
        
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj = self.relation_mlp_model[relation_name](x_combined)  # [B, N*O, 1]
        adj = adj.reshape(B, N1, N2)  # [B, N, O]
        
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 

    def build_ss_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  self.alpha*(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 


    def build_graph(self, spatial_nodes, temporal_nodes):
        
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) 
        ss_adj = self.build_sub_graph(spatial_nodes, spatial_nodes,'ss' )
        st_adj  = self.build_sub_graph(spatial_nodes, temporal_nodes,'st' )
        ts_adj = self.build_sub_graph(temporal_nodes, spatial_nodes,'ts' )
        tt_adj = adj[:, N:, N:]
        adj[:, :N, N:] = st_adj
        adj[:, N:, :N] = ts_adj
        adj[:, :N, :N] = ss_adj
        
        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            mask =  self.predefined_adj[:N, :N] != 0
            result = self.predefined_adj[:N, :N] * mask
            ss_adj = ss_adj * result
            # ss_adj = ss_adj + self.predefined_adj[:N, :N]
            
        ss_adj=   self.uni_adj(ss_adj)
        st_adj =  self.uni_adj(st_adj) #torch.tanh( torch.relu(st_adj))# * F.softmax(st_adj, dim=1)
        ts_adj =  self.uni_adj(ts_adj) #torch.tanh( torch.relu( ts_adj)) #* F.softmax(ts_adj, dim=1)
        tt_adj = self.uni_adj(tt_adj) #torch.tanh( torch.relu(tt_adj) / torch.max(torch.relu(tt_adj))  ) #* F.softmax(tt_adj, dim=1)
        
        adj[:, :N, :N]=  ss_adj#/torch.max(ss_adj)   #* F.softmax(ss_adj, dim=1)
        adj[:, :N, N:] = st_adj#/torch.max(st_adj)# * F.softmax(st_adj, dim=1)
        adj[:, N:, :N] = ts_adj#/torch.max(ts_adj)   #* F.softmax(ts_adj, dim=1)
        adj[:, N:, N:]  = tt_adj#/torch.max(tt_adj)  #* F.softmax(tt_adj, dim=1)
        
        if self.tt_mask:
            adj[:, N:, N:] = torch.triu(adj[:, N:, N:])


        return adj
        
    def uni_adj(self, adj):
        epsilon = 1e-30  # 一个小常数，防止数值不稳定
        div = torch.max(torch.relu(adj))
        if div == 0:
            adj_processed = torch.tanh(torch.relu(adj))
        else:
            adj_processed = torch.tanh(torch.relu(adj) / (div + epsilon))  # 加入小常数 epsilon 防止数值不稳定
        return torch.nan_to_num(adj_processed)  # 处理可能的 nan 值



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        spatial_nodes = spatial_nodes.detach()
        temporal_nodes = temporal_nodes.detach()

        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        adj = self.build_graph(spatial_nodes, temporal_nodes)
                # predefined adjcent matrix        


        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values



class MLPSimSampleDirectConstructor(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,tt_mask=True, alpha=3):
        super(MLPSimSampleDirectConstructor, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        self.tt_mask = tt_mask
        self.relation_mlp_model = nn.ModuleDict({
            'st': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, 1),
                ),
            'ts': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, 1),
                ),
        })
        self.alpha = alpha
        
        
        self.ss_x1_lin = nn.Linear(latent_dim, latent_dim, bias=False)
        self.ss_x2_lin = nn.Linear(latent_dim, latent_dim, bias=False)
    
    def build_sub_graph(self, x1, x2, relation_name):
        
        if relation_name =='ss':
            adj = self.build_ss_graph(x1, x2)
            return adj
        
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs   xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj = self.relation_mlp_model[relation_name](x_combined)  # [B, N*O, 1]
        adj = adj.reshape(B, N1, N2)  # [B, N, O]
        
        return adj 

    def build_ss_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  torch.tanh(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 

    def sample_sub_adj(self, sub_adj):
        adj_logits = torch.sigmoid(sub_adj)
        adj_logits_ = (adj_logits / torch.max(adj_logits))
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=0.2, probs=adj_logits_).rsample()
    
        return adj_sampled*adj_logits_
    
    def build_graph(self, spatial_nodes, temporal_nodes):
        
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) 
        ss_adj = self.build_sub_graph(spatial_nodes, spatial_nodes,'ss' )
        st_adj  = self.build_sub_graph(spatial_nodes, temporal_nodes,'st' )
        ts_adj = self.build_sub_graph(temporal_nodes, spatial_nodes,'ts' )
        tt_adj = adj[:, N:, N:]
        
        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            adj = adj + self.predefined_adj

        
        adj[:, :N, N:] = self.sample_sub_adj(st_adj)
        adj[:, N:, :N] = self.sample_sub_adj(ts_adj)
        adj[:, N:, N:] = self.sample_sub_adj(tt_adj)
        adj[:, :N, :N] = self.sample_sub_adj(ss_adj)
        

        
        
        if self.tt_mask:
            adj[:, N:, N:] = torch.triu(adj[:, N:, N:])
        
        return adj



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        spatial_nodes = spatial_nodes.detach()
        temporal_nodes = temporal_nodes.detach()

        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        _adj = self.build_graph(spatial_nodes, temporal_nodes)
        # adj = _adj * mask
        adj = torch.tanh(torch.relu(_adj))

        # predefined adjcent matrix        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            adj = adj + self.predefined_adj
        

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values













class MLPSimDirectConstructor(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,tt_mask=True, alpha=3):
        super(MLPSimDirectConstructor, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        self.tt_mask = tt_mask
        self.relation_mlp_model = nn.ModuleDict({
            'st': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
            'ts': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
        })
        
        
                
        self.relation_mlp_model = nn.ModuleDict({
            'ss': nn.Linear(latent_dim+latent_dim, 1),
            'tt': nn.Linear(latent_dim+latent_dim, 1),
            'st': nn.Linear(latent_dim+latent_dim, 1),
            'ts': nn.Linear(latent_dim+latent_dim, 1),
        })
        self.mask_mlp_model = nn.ModuleDict({
            'ss': nn.Linear(latent_dim+latent_dim, 1),
            'tt': nn.Linear(latent_dim+latent_dim, 1),
            'st': nn.Linear(latent_dim+latent_dim, 1),
            'ts': nn.Linear(latent_dim+latent_dim, 1),
        })

        self.alpha = alpha
        
        
        self.ss_x1_lin = nn.Linear(latent_dim, latent_dim, bias=False)
        self.ss_x2_lin = nn.Linear(latent_dim, latent_dim, bias=False)
    
    def build_sub_graph(self, x1, x2, relation_name):
        
        if relation_name =='ss':
            return self.build_ss_graph(x1, x2)
        
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj = self.relation_mlp_model[relation_name](x_combined)  # [B, N*O, 1]
        adj = adj.reshape(B, N1, N2)  # [B, N, O]
        
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 

    def build_ss_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  torch.tanh(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 


    def build_graph(self, spatial_nodes, temporal_nodes):
        
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) 
        ss_adj = self.build_sub_graph(spatial_nodes, spatial_nodes,'ss' )
        st_adj  = self.build_sub_graph(spatial_nodes, temporal_nodes,'st' )
        ts_adj = self.build_sub_graph(temporal_nodes, spatial_nodes,'ts' )
        adj[:, :N, N:] = st_adj
        adj[:, N:, :N] = ts_adj
        adj[:, :N, :N] = ss_adj
        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            adj = adj + self.predefined_adj
        
        if self.tt_mask:
            adj[:, N:, N:] = torch.triu(adj[:, N:, N:])
        
        return adj



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        spatial_nodes = spatial_nodes.detach()
        temporal_nodes = temporal_nodes.detach()

        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        _adj = self.build_graph(spatial_nodes, temporal_nodes)
                # predefined adjcent matrix        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            _adj = _adj + self.predefined_adj
        

        # adj = _adj * mask
        adj = torch.tanh(torch.relu(_adj))


        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values




import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, dim):
        super(AttentionMechanism, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


    def forward(self, x1, x2):
        # Generate query, key, value
        q = self.query(x1)  # [B, N1, D]
        k = self.key(x2)    # [B, N2, D]
        # v = self.value(x2)  # [B, N2, D]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, N1, N2]
        # attn_scores = F.softmax(attn_scores, dim=1)

        # Apply attention to the values
        # attn_output = torch.matmul(attn_scores, v)  # [B, N1, D]

        return attn_scores





class AttentionSimDirectConstructor(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,tt_mask=True, alpha=3, N=0, O=0):
        super(AttentionSimDirectConstructor, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        

        self.relation_mlp_model = nn.ModuleDict({
            'st': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
            'ts': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
        })


        self.tt_mask = tt_mask
        self.relation_attention_model = nn.ModuleDict({
            'st': AttentionMechanism(latent_dim),
            'ts': AttentionMechanism(latent_dim),
        })
        
        
                
        # self.relation_mask_model = nn.ModuleDict({
        #     'st': nn.Sequential(
        #             nn.Linear(latent_dim+latent_dim, latent_dim),
        #             nn.ELU(),
        #             nn.LayerNorm(latent_dim),
        #             nn.Linear(latent_dim, 1),
        #         ),
        #     'ts': nn.Sequential(
        #             nn.Linear(latent_dim+latent_dim, latent_dim),
        #             nn.ELU(),
        #             nn.LayerNorm(latent_dim),
        #             nn.Linear(latent_dim, 1),
        #         ),
        # })
        self.relation_mlp_model = nn.ModuleDict({
            'st': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
            'ts': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
        })
        self.alpha = alpha
        
        
        self.ss_x1_lin = nn.Linear(latent_dim, latent_dim, bias=False)
        self.ss_x2_lin = nn.Linear(latent_dim, latent_dim, bias=False)
    
    def build_sub_graph(self, x1, x2, relation_name):
        
        if relation_name =='ss':
            return self.build_ss_graph(x1, x2)
        
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj_  = self.relation_mlp_model[relation_name](x_combined)
        adj_ = adj_.reshape(B, N1, N2)  # [B, N, O]
        
        return adj_ 

    def build_tt_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  torch.tanh(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 

    def build_ss_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1)
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 
    
    def uni_adj(self, adj):
        epsilon = 1e-30  # 一个小常数，防止数值不稳定
        adj = torch.clamp(adj, min=0, max=1000)  # 限制 adj 的值范围以避免梯度爆炸
        div = torch.max(torch.relu(adj))
        if div == 0:
            adj_processed = torch.tanh(torch.relu(adj))
        else:
            adj_processed = torch.tanh(torch.relu(adj) / (div + epsilon))  # 加入小常数 epsilon 防止数值不稳定
        return torch.nan_to_num(adj_processed)  # 处理可能的 nan 值
    
    def build_graph(self, spatial_nodes, temporal_nodes):
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) 
        ss_adj = self.build_ss_graph(spatial_nodes, spatial_nodes)
        st_adj  = adj[:, :N, N:] + self.build_sub_graph(spatial_nodes, temporal_nodes , 'st')
        ts_adj =  adj[:, N:, :N] + self.build_sub_graph(temporal_nodes, spatial_nodes, 'ts')
        tt_adj = adj[:, N:, N:]
        
                
        if self.predefined_adj is not None:
            # avoid inplace operation 
            adj = adj + self.predefined_adj

        # adj[:, :N, :N]= torch.tanh( torch.relu(ss_adj)/ss_adj) * F.softmax(ss_adj, dim=1)
        # adj[:, :N, N:] = torch.tanh( torch.relu(st_adj)/st_adj) * F.softmax(st_adj, dim=1)
        # adj[:, N:, :N] = torch.tanh( torch.relu(ts_adj)/ts_adj)* F.softmax(ts_adj, dim=1)
        # adj[:, N:, N:]  = torch.tanh( torch.relu(tt_adj)/tt_adj) * F.softmax(tt_adj, dim=1)
        ss_adj=   self.uni_adj(ss_adj) #torch.tanh( torch.relu(ss_adj))   #* F.softmax(ss_adj, dim=1)
        st_adj =  self.uni_adj(st_adj) #torch.tanh( torch.relu(st_adj))# * F.softmax(st_adj, dim=1)
        ts_adj =  self.uni_adj(ts_adj) #torch.tanh( torch.relu( ts_adj)) #* F.softmax(ts_adj, dim=1)
        tt_adj = self.uni_adj(tt_adj) #torch.tanh( torch.relu(tt_adj) / torch.max(torch.relu(tt_adj))  ) #* F.softmax(tt_adj, dim=1)
        
        
        

        adj[:, :N, :N]=  ss_adj#/torch.max(ss_adj)   #* F.softmax(ss_adj, dim=1)
        adj[:, :N, N:] = st_adj#/torch.max(st_adj)# * F.softmax(st_adj, dim=1)
        adj[:, N:, :N] = ts_adj#/torch.max(ts_adj)   #* F.softmax(ts_adj, dim=1)
        adj[:, N:, N:]  = tt_adj#/torch.max(tt_adj)  #* F.softmax(tt_adj, dim=1)

        return adj



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        spatial_nodes = spatial_nodes.detach()
        temporal_nodes = temporal_nodes.detach()

        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        adj = self.build_graph(spatial_nodes, temporal_nodes)
                # predefined adjcent matrix        

        # adj = _adj * mask
        # adj = torch.tanh( torch.sqrt(torch.relu(_adj) * F.softmax(_adj, dim=-1)))

        if self.tt_mask:
            adj[:, N:, N:] = torch.triu(adj[:, N:, N:])

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values









class AttentionSimDirectConstructor2(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,tt_mask=True, alpha=3, N=0, O=0):
        super(AttentionSimDirectConstructor2, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        

        # self.relation_mlp_model = nn.ModuleDict({
        #     'st': nn.Sequential(
        #             nn.Linear(latent_dim+latent_dim, latent_dim),
        #             nn.ELU(),
        #             nn.Linear(latent_dim, 1),
        #         ),
        #     'ts': nn.Sequential(
        #             nn.Linear(latent_dim+latent_dim, latent_dim),
        #             nn.ELU(),
        #             nn.Linear(latent_dim, 1),
        #         ),
        # })
        self.relation_attention_model = nn.ModuleDict({
            'st': AttentionMechanism(latent_dim),
            'ts': AttentionMechanism(latent_dim),
        })


        self.tt_mask = tt_mask
        self.alpha = alpha
        
        
        self.ss_x1_lin = nn.Linear(latent_dim, latent_dim, bias=False)
        self.ss_x2_lin = nn.Linear(latent_dim, latent_dim, bias=False)
    
    def build_sub_graph(self, x1, x2, relation_name):
        
        if relation_name =='ss':
            return self.build_ss_graph(x1, x2)
        
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj_  = self.relation_attention_model[relation_name](x1, x2)
        adj_ = adj_.reshape(B, N1, N2)  # [B, N, O]
        
        return adj_ 

    def build_tt_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  torch.tanh(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 

    def build_ss_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  self.alpha*(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 
    
    def uni_adj(self, adj):
        epsilon = 1e-30  # 一个小常数，防止数值不稳定
        div = torch.max(torch.relu(adj))
        if div == 0:
            adj_processed = torch.tanh(torch.relu(adj))
        else:
            adj_processed = torch.tanh(torch.relu(adj) / (div + epsilon))  # 加入小常数 epsilon 防止数值不稳定
        return torch.nan_to_num(adj_processed)  # 处理可能的 nan 值
    
    def build_graph(self, spatial_nodes, temporal_nodes):
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) 
        ss_adj = adj[:, :N, :N] + self.build_ss_graph(spatial_nodes, spatial_nodes) 
        st_adj  = adj[:, :N, N:] + self.build_sub_graph(spatial_nodes, temporal_nodes , 'st')
        ts_adj =  adj[:, N:, :N] + self.build_sub_graph(temporal_nodes, spatial_nodes, 'ts')
        tt_adj = adj[:, N:, N:]
        

        # adj[:, :N, :N]= torch.tanh( torch.relu(ss_adj)/ss_adj) * F.softmax(ss_adj, dim=1)
        # adj[:, :N, N:] = torch.tanh( torch.relu(st_adj)/st_adj) * F.softmax(st_adj, dim=1)
        # adj[:, N:, :N] = torch.tanh( torch.relu(ts_adj)/ts_adj)* F.softmax(ts_adj, dim=1)
        # adj[:, N:, N:]  = torch.tanh( torch.relu(tt_adj)/tt_adj) * F.softmax(tt_adj, dim=1)
        # ss_adj=   self.uni_adj(ss_adj) *  #torch.tanh( torch.relu(ss_adj))   #* F.softmax(ss_adj, dim=1)
        
        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            mask =  self.predefined_adj[:N, :N] != 0
            result = self.predefined_adj[:N, :N] * mask
            ss_adj = ss_adj * result
            # ss_adj = ss_adj + self.predefined_adj[:N, :N]
            
        ss_adj=   self.uni_adj(ss_adj)
        st_adj =  self.uni_adj(st_adj) #torch.tanh( torch.relu(st_adj))# * F.softmax(st_adj, dim=1)
        ts_adj =  self.uni_adj(ts_adj) #torch.tanh( torch.relu( ts_adj)) #* F.softmax(ts_adj, dim=1)
        tt_adj = self.uni_adj(tt_adj) #torch.tanh( torch.relu(tt_adj) / torch.max(torch.relu(tt_adj))  ) #* F.softmax(tt_adj, dim=1)
        

        adj[:, :N, :N]=  ss_adj#/torch.max(ss_adj)   #* F.softmax(ss_adj, dim=1)
        adj[:, :N, N:] = st_adj#/torch.max(st_adj)# * F.softmax(st_adj, dim=1)
        adj[:, N:, :N] = ts_adj#/torch.max(ts_adj)   #* F.softmax(ts_adj, dim=1)
        adj[:, N:, N:]  = tt_adj#/torch.max(tt_adj)  #* F.softmax(tt_adj, dim=1)

        return adj



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        spatial_nodes = spatial_nodes.detach()
        temporal_nodes = temporal_nodes.detach()

        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        adj = self.build_graph(spatial_nodes, temporal_nodes)
                # predefined adjcent matrix        

        # adj = _adj * mask
        # adj = torch.tanh( torch.sqrt(torch.relu(_adj) * F.softmax(_adj, dim=-1)))

        if self.tt_mask:
            adj[:, N:, N:] = torch.triu(adj[:, N:, N:])

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values


















class AttentionSimDirectConstructor3(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,tt_mask=True, alpha=3, N=0, O=0):
        super(AttentionSimDirectConstructor3, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        

        # self.relation_mlp_model = nn.ModuleDict({
        #     'st': nn.Sequential(
        #             nn.Linear(latent_dim+latent_dim, latent_dim),
        #             nn.ELU(),
        #             nn.Linear(latent_dim, 1),
        #         ),
        #     'ts': nn.Sequential(
        #             nn.Linear(latent_dim+latent_dim, latent_dim),
        #             nn.ELU(),
        #             nn.Linear(latent_dim, 1),
        #         ),
        # })
        self.relation_attention_model = nn.ModuleDict({
            'st': AttentionMechanism(latent_dim),
            'ts': AttentionMechanism(latent_dim),
        })


        self.tt_mask = tt_mask
        self.alpha = alpha
        
        
        self.ss_x1_lin = nn.Linear(latent_dim, latent_dim, bias=False)
        self.ss_x2_lin = nn.Linear(latent_dim, latent_dim, bias=False)
    
    def build_sub_graph(self, x1, x2, relation_name):
        
        if relation_name =='ss':
            return self.build_ss_graph(x1, x2)
        
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj_  = self.relation_attention_model[relation_name](x1, x2)
        adj_ = adj_.reshape(B, N1, N2)  # [B, N, O]
        
        return adj_ 

    def build_tt_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  torch.tanh(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 

    def build_ss_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  self.alpha*(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 
    
    def uni_adj(self, adj):
        epsilon = 1e-30  # 一个小常数，防止数值不稳定
        div = torch.max(torch.relu(adj))
        if div == 0:
            adj_processed = torch.tanh(torch.relu(adj))
        else:
            adj_processed = torch.tanh(torch.relu(adj) / (div + epsilon))  # 加入小常数 epsilon 防止数值不稳定
        return torch.nan_to_num(adj_processed)  # 处理可能的 nan 值
    
    def build_graph(self, spatial_nodes, temporal_nodes):
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) 
        ss_adj = adj[:, :N, :N] + self.build_ss_graph(spatial_nodes, spatial_nodes) 
        st_adj  = adj[:, :N, N:] + self.build_sub_graph(spatial_nodes, temporal_nodes , 'st')
        ts_adj =  adj[:, N:, :N] + self.build_sub_graph(temporal_nodes, spatial_nodes, 'ts')
        tt_adj = adj[:, N:, N:]
        

        # adj[:, :N, :N]= torch.tanh( torch.relu(ss_adj)/ss_adj) * F.softmax(ss_adj, dim=1)
        # adj[:, :N, N:] = torch.tanh( torch.relu(st_adj)/st_adj) * F.softmax(st_adj, dim=1)
        # adj[:, N:, :N] = torch.tanh( torch.relu(ts_adj)/ts_adj)* F.softmax(ts_adj, dim=1)
        # adj[:, N:, N:]  = torch.tanh( torch.relu(tt_adj)/tt_adj) * F.softmax(tt_adj, dim=1)
        # ss_adj=   self.uni_adj(ss_adj) *  #torch.tanh( torch.relu(ss_adj))   #* F.softmax(ss_adj, dim=1)
        
        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            mask =  self.predefined_adj[:N, :N] != 0
            result = self.predefined_adj[:N, :N] * mask
            ss_adj = ss_adj * result
            # ss_adj = ss_adj + self.predefined_adj[:N, :N]
            
        ss_adj=   self.uni_adj(ss_adj)
        st_adj =  self.uni_adj(st_adj) #torch.tanh( torch.relu(st_adj))# * F.softmax(st_adj, dim=1)
        ts_adj =  self.uni_adj(ts_adj) #torch.tanh( torch.relu( ts_adj)) #* F.softmax(ts_adj, dim=1)
        tt_adj = self.uni_adj(tt_adj) #torch.tanh( torch.relu(tt_adj) / torch.max(torch.relu(tt_adj))  ) #* F.softmax(tt_adj, dim=1)
        

        adj[:, :N, :N]=  ss_adj#/torch.max(ss_adj)   #* F.softmax(ss_adj, dim=1)
        adj[:, :N, N:] = st_adj#/torch.max(st_adj)# * F.softmax(st_adj, dim=1)
        adj[:, N:, :N] = ts_adj#/torch.max(ts_adj)   #* F.softmax(ts_adj, dim=1)
        adj[:, N:, N:]  = tt_adj#/torch.max(tt_adj)  #* F.softmax(tt_adj, dim=1)

        return adj



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        spatial_nodes = spatial_nodes.detach()
        temporal_nodes = temporal_nodes.detach()

        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        adj = self.build_graph(spatial_nodes, temporal_nodes)
                # predefined adjcent matrix        

        # adj = _adj * mask
        # adj = torch.tanh( torch.sqrt(torch.relu(_adj) * F.softmax(_adj, dim=-1)))

        if self.tt_mask:
            adj[:, N:, N:] = torch.triu(adj[:, N:, N:])

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values










class AttentionSimDirectConstructor4(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,tt_mask=True, alpha=3, N=0, O=0):
        super(AttentionSimDirectConstructor4, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        

        self.relation_mlp_model = nn.ModuleDict({
            'st': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
            'ts': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.ELU(),
                    nn.Linear(latent_dim, 1),
                ),
        })
        # self.relation_attention_model = nn.ModuleDict({
        #     'st': AttentionMechanism(latent_dim),
        #     'ts': AttentionMechanism(latent_dim),
        # })


        self.tt_mask = tt_mask
        self.alpha = alpha
        
        
        self.ss_x1_lin = nn.Linear(latent_dim, latent_dim, bias=False)
        self.ss_x2_lin = nn.Linear(latent_dim, latent_dim, bias=False)
    
    def build_sub_graph(self, x1, x2, relation_name):
        
        if relation_name =='ss':
            return self.build_ss_graph(x1, x2)
        
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        # 扩展 x1 和 x2 以便于拼接
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        # 在拼接维度上重复以匹配对方的维度
        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        # 拼接 xs 和 xt
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
        # 重塑并应用 MLP
        x_combined = x_combined.reshape(B, N1 * N2, -1)  # [B, N*O, 2D]
        
        adj_  = self.relation_mlp_model[relation_name](x_combined)
        adj_ = adj_.reshape(B, N1, N2)  # [B, N, O]
        
        return adj_ 

    def build_tt_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  torch.tanh(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 

    def build_ss_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  self.alpha*(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
        # adj = torch.relu(torch.tanh(self.alpha*a))
        # output = self.mlp_output(xi)  # (B, N, O)
        return adj 
    
    def uni_adj(self, adj):
        epsilon = 1e-30  # 一个小常数，防止数值不稳定
        div = torch.max(torch.relu(adj))
        if div == 0:
            adj_processed = torch.tanh(torch.relu(adj))
        else:
            adj_processed = torch.tanh(torch.relu(adj) / (div + epsilon))  # 加入小常数 epsilon 防止数值不稳定
        return torch.nan_to_num(adj_processed)  # 处理可能的 nan 值
    
    def build_graph(self, spatial_nodes, temporal_nodes):
        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        node_embs = torch.concat([spatial_nodes, temporal_nodes], dim=1)
        adj = torch.einsum("bnf, bmf -> bnm", node_embs, node_embs) 
        ss_adj = adj[:, :N, :N] + self.build_ss_graph(spatial_nodes, spatial_nodes) 
        st_adj  = adj[:, :N, N:] + self.build_sub_graph(spatial_nodes, temporal_nodes , 'st')
        ts_adj =  adj[:, N:, :N] + self.build_sub_graph(temporal_nodes, spatial_nodes, 'ts')
        tt_adj = adj[:, N:, N:]
        

        # adj[:, :N, :N]= torch.tanh( torch.relu(ss_adj)/ss_adj) * F.softmax(ss_adj, dim=1)
        # adj[:, :N, N:] = torch.tanh( torch.relu(st_adj)/st_adj) * F.softmax(st_adj, dim=1)
        # adj[:, N:, :N] = torch.tanh( torch.relu(ts_adj)/ts_adj)* F.softmax(ts_adj, dim=1)
        # adj[:, N:, N:]  = torch.tanh( torch.relu(tt_adj)/tt_adj) * F.softmax(tt_adj, dim=1)
        # ss_adj=   self.uni_adj(ss_adj) *  #torch.tanh( torch.relu(ss_adj))   #* F.softmax(ss_adj, dim=1)
        
        
        if self.predefined_adj is not None:
            # avoid inplace operation 
            mask =  self.predefined_adj[:N, :N] != 0
            result = self.predefined_adj[:N, :N] * mask
            ss_adj = ss_adj * result
            # ss_adj = ss_adj + self.predefined_adj[:N, :N]
            
        ss_adj=   self.uni_adj(ss_adj)
        st_adj =  self.uni_adj(st_adj) #torch.tanh( torch.relu(st_adj))# * F.softmax(st_adj, dim=1)
        ts_adj =  self.uni_adj(ts_adj) #torch.tanh( torch.relu( ts_adj)) #* F.softmax(ts_adj, dim=1)
        tt_adj = self.uni_adj(tt_adj) #torch.tanh( torch.relu(tt_adj) / torch.max(torch.relu(tt_adj))  ) #* F.softmax(tt_adj, dim=1)
        

        adj[:, :N, :N]=  ss_adj#/torch.max(ss_adj)   #* F.softmax(ss_adj, dim=1)
        adj[:, :N, N:] = st_adj#/torch.max(st_adj)# * F.softmax(st_adj, dim=1)
        adj[:, N:, :N] = ts_adj#/torch.max(ts_adj)   #* F.softmax(ts_adj, dim=1)
        adj[:, N:, N:]  = tt_adj#/torch.max(tt_adj)  #* F.softmax(tt_adj, dim=1)

        return adj



    def forward(
        self, spatial_nodes, temporal_nodes
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # spatial nodes: (B, N, D)
        # temporal nodes: (B, T, D)
        # A : (N,T), (T , N)
        spatial_nodes = spatial_nodes.detach()
        temporal_nodes = temporal_nodes.detach()

        B, N, D = spatial_nodes.size()
        _, T, _ = temporal_nodes.size()
        
        adj = self.build_graph(spatial_nodes, temporal_nodes)
                # predefined adjcent matrix        

        # adj = _adj * mask
        # adj = torch.tanh( torch.sqrt(torch.relu(_adj) * F.softmax(_adj, dim=-1)))

        if self.tt_mask:
            adj[:, N:, N:] = torch.triu(adj[:, N:, N:])

        batch_indices = list()
        batch_values = list()
        for bi in range(B):
            source_nodes, target_nodes = adj[bi].nonzero().t()
            edge_weights = adj[bi][source_nodes, target_nodes]
            edge_index_i = torch.stack([source_nodes, target_nodes], dim=0)
            batch_indices.append(edge_index_i)
            batch_values.append(edge_weights)
        return adj, batch_indices, batch_values


















