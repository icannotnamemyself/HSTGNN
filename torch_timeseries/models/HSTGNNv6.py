from typing import List, Tuple
import torch
from torch import nn
from torch_timeseries.layers.heterostgcn5 import HeteroFASTGCN as HeteroFASTGCN5
from torch_timeseries.layers.heterostgcn6 import HeteroFASTGCN as HeteroFASTGCN6
from torch_timeseries.layers.han import HAN
from torch_timeseries.layers.hgt import HGT
from torch_timeseries.layers.nlinear_output import NlinearOuputLayer
from torch_timeseries.layers.weighted_han import WeightedHAN
from torch_timeseries.layers.weighted_han_update import WeightedHAN as WeightedHANUpdate
from torch_timeseries.layers.weighted_han_update2 import WeightedHAN as WeightedHANUpdate2
from torch_timeseries.layers.weighted_han_update3 import WeightedHAN as WeightedHANUpdate3
from torch_timeseries.layers.weighted_han_update4 import WeightedHAN as WeightedHANUpdate4
from torch_timeseries.layers.weighted_han_update5 import WeightedHAN as WeightedHANUpdate5
from torch_timeseries.layers.graphsage import MyGraphSage, MyFAGCN
from torch_timeseries.nn.egsage import EGraphSage
from torch_timeseries.nn.herero_esage import GNNStack, HeteroSTEGraphSage


class HSTGNN(nn.Module):
    def __init__(
        self,
        seq_len,
        num_nodes,
        temporal_embed_dim,
        graph_build_type="mlpsim", # "predefined_adaptive"
        graph_conv_type="hstga1",
        output_layer_type='tcn8',
        heads=1,
        negative_slope=0.2,
        gcn_layers=2,
        rebuild_time=True,
        rebuild_space=True,
        rebuild_nss=True,
        rebuild_ntt=True,
        rebuild_nts=True,
        rebuild_nst=True,
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
        without_gcn=False,
        d0=2,
        gnn_layer_num=3,
        kernel_set=[2,3,6,7],
        normalization=True,
        conv_type='all',
        device='cuda'
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
        self.gnn_layer_num = gnn_layer_num
        # self.spatial_encoder = SpatialEncoder(
        #     seq_len,
        #     num_nodes,
        #     static_embed_dim=node_static_embed_dim,
        #     latent_dim=latent_dim,
        # )
        # self.temporal_encoder = TemporalEncoder(
        #     seq_len,
        #     num_nodes,
        #     temporal_embed_dim,
        #     static_embed_dim=node_static_embed_dim,
        #     latent_dim=latent_dim,
        # )
        
        self.embed_convs = GNNStack(self.num_nodes, 1, latent_dim, latent_dim, 0.05, 'elu',gnn_layer_num, self.num_nodes)


        self.hot_encoding_matrix = torch.eye(num_nodes).detach()
        self.temporal_encoding_matrix = torch.arange(0, seq_len) / seq_len # 
        self.temporal_encoding_matrix = self.temporal_encoding_matrix.view(-1,1).expand(-1, num_nodes).detach()

        row_indices = torch.arange(num_nodes).repeat_interleave(seq_len)
        col_indices = torch.arange(num_nodes, num_nodes + seq_len).repeat(num_nodes) 
        edge_index = torch.stack([row_indices, col_indices], dim=0).detach()
        # bi_edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)
        self.sage_edge_index= torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1).to(device).detach()
        # 复制 edge_attr 以匹配双向边
        
        self.x_input = torch.concat([self.hot_encoding_matrix, self.temporal_encoding_matrix], dim=0).to(device).detach()
        


        self.rebuild_time = rebuild_time
        self.rebuild_space = rebuild_space
        self.rebuild_nss = rebuild_nss
        self.rebuild_ntt = rebuild_ntt
        self.rebuild_nst = rebuild_nst
        self.rebuild_nts = rebuild_nts
        
        # Rebuild Module
        if rebuild_space:
            self.feature_rebuild = nn.Sequential(
                nn.Linear(latent_dim*3, seq_len),
                torch.nn.ELU(),
                nn.Linear(seq_len, seq_len),
            )

        if rebuild_time:
            self.time_rebuild = nn.GRU(latent_dim*3, num_nodes, batch_first=True)
        
        if rebuild_nss:
            self.nss_rebuild = nn.Sequential(
                nn.Linear(latent_dim, seq_len),
                torch.nn.ELU(),
                nn.Linear(seq_len, seq_len),
            )
        if rebuild_ntt:
            self.ntt_rebuild = nn.GRU(latent_dim, num_nodes, batch_first=True)

        if rebuild_nst:
            self.nst_rebuild = nn.GRU(latent_dim, num_nodes, batch_first=True)
            
        if rebuild_nts:
            self.nts_rebuild = nn.Sequential(
                nn.Linear(latent_dim, seq_len),
                torch.nn.ELU(),
                nn.Linear(seq_len, seq_len),
            )
            
            
        self.activation = torch.nn.GELU()
            
        
        
        
        self.tn_modules = nn.ModuleList()
        
        if not self.without_tn_module:
            for i in range(1):
                self.tn_modules.append(
                    TNModule(
                        num_nodes,
                        seq_len,
                        latent_dim*3,
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
        # if rebuild_nss:
        #     out_channels = out_channels + 1
        # if rebuild_ntt:
        #     out_channels = out_channels + 1
        # if rebuild_nst:
        #     out_channels = out_channels + 1
        # if rebuild_nts:
        #     out_channels = out_channels + 1

        
        if output_layer_type == 'tcn8':
            self.output_layer = TCNOutputLayer(
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


    def build_x_embed(self, x):
        pass
    
    
    
        
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
        

        (B, N, T) = x.size()

        x_embed = []
        for i in range(B):
            xi = x[i]
            

            # 创建 edge_attr
            edge_attr = xi.flatten()
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0).unsqueeze(-1)
            # 转换为 PyTorch 张量
            # edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            # edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

            xc = self.embed_convs(self.x_input, edge_attr, self.sage_edge_index)
            x_embed.append(xc)
        x_embed = torch.stack(x_embed, dim=0) # (B, N+T, D)
        
        if not self.without_tn_module:
            for i in range(1):
                X, (nss, ntt, nst, nts), (smt_s, smt_t) = self.tn_modules[i](x_embed)  # (B, N+T, latent_dim)

        # rebuild module
        Xs = X[:, : self.num_nodes, :]  # (B, N, D)
        Xt = X[:, self.num_nodes :, :]  # (B, T, D)
        
        
        
        outputs = list()
        if self.rebuild_space:
            n_output = self.feature_rebuild(Xs)   # (B, N, T)
            outputs.append(n_output.unsqueeze(1))
        if self.rebuild_time:
            t_output,_ = self.time_rebuild(Xt)  # (B, T, N)
            outputs.append(t_output.unsqueeze(1).transpose(2, 3))
        
        # if self.rebuild_nss:
        #     nss_output = self.nss_rebuild(nss)   # (B, N, T)
        #     outputs.append(nss_output.unsqueeze(1))
        # if self.rebuild_nst:
        #     nst_output, _ = self.nst_rebuild(nst)  # (B, T, N)
        #     outputs.append(nst_output.unsqueeze(1).transpose(2, 3))
        # if self.rebuild_nts:
        #     nts_output = self.nts_rebuild(nts)   # (B, N, T)
        #     outputs.append(nts_output.unsqueeze(1))
        # if self.rebuild_ntt:
        #     ntt_output, _  = self.ntt_rebuild(ntt)  # (B, T, N)
        #     outputs.append(ntt_output.unsqueeze(1).transpose(2, 3))

        outputs.append(x.unsqueeze(1))  # （B, 1/2/3, N, T)

        # output module
        X = self.activation(torch.cat(outputs, dim=1))  # （B, 1/2/3, N, T)
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
            print("graph_build_type is mlp2")

        elif graph_build_type == "mlpsim":
            self.graph_constructor = MLPSimConstructor(predefined_adj=predefined_adj,latent_dim=latent_dim,self_loop_eps=self.self_loop_eps)
            print("graph_build_type is mlpsim")


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
                    num_nodes, seq_len, 3*latent_dim, latent_dim,latent_dim, gcn_layers,
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
            elif graph_conv_type == 'hgt':
                self.graph_conv = HGT(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type='all'
                )
            elif graph_conv_type == 'hgt':
                self.graph_conv = HGT(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type='all'
                )
            elif graph_conv_type == 'hstga1':
                self.graph_conv = HSTGAttn(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                conv_type=self.conv_type
                )


            else:
                raise NotImplementedError("Unknown graph_conv", graph_conv_type)

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
             X, (nss, ntt, nst, nts), (smt_s, smt_t) = self.graph_conv(X, batch_indices, batch_values)

        return X, (nss, ntt, nst, nts), (smt_s, smt_t)


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
            torch.nn.Sigmoid()
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
    def __init__(self, predefined_adj=None, latent_dim=32,self_loop_eps=0.5):
        super(MLPSimConstructor, self).__init__()
        self.predefined_adj =predefined_adj
        # self.self_loop_eps = self_loop_eps
        
        self.relation_mlp_model = nn.ModuleDict({
            'st': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.Sigmoid(),
                    nn.Linear(latent_dim, 1),
                ),
            'ts': nn.Sequential(
                    nn.Linear(latent_dim+latent_dim, latent_dim),
                    nn.Sigmoid(),
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







import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.dilated_inception import DilatedInception
from torch_timeseries.nn.layer import LayerNorm


class TCNOutputLayer(nn.Module):
    def __init__(self,input_seq_len,num_nodes,out_seq_len,tcn_layers,in_channel,dilated_factor,tcn_channel, hidden_channel=3,kernel_set=[2,3,6,7],d0=1) -> None:
        super().__init__()
        
        # self.latent_seq_layer = nn.Linear(input_seq_len, latent_seq)
        # self.hidden_layer = nn.Conv2d(in_channel, hidden_channel, (1,1))
        self.channel_layer = nn.Conv2d(in_channel, tcn_channel, (1, 1))
        self.tcn = TCN(
                    input_seq_len,num_nodes, tcn_channel, tcn_channel,
                    out_seq_len=out_seq_len, num_layers=tcn_layers,
                    dilated_factor=dilated_factor, d0=d0,kernel_set=kernel_set
                )
        self.end_layer = nn.Conv2d(tcn_channel, out_seq_len, (1, 1))
        self.act = nn.ELU()
        
    def forward(self, x):
        # output = self.act(self.latent_seq_layer(x))# (B ,C , N, latent_seq)
        # output = self.act(self.hidden_layer(x)) 
        output = self.act(self.channel_layer(x))  # (B ,C , N, T)
        output = self.tcn(output)  # (B, C, N, out_len)
        output = self.end_layer(output).squeeze(3)  # (B, out_len, N)
        return output




class TCN(nn.Module):
    def __init__(
        self, seq_len,num_nodes, in_channels, hidden_channels, out_seq_len=1, out_channels=None, num_layers=5, 
        d0=1,dilated_factor=2,  dropout=0,kernel_set=[2,3,6,7]
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dropout = dropout
        self.act = nn.ReLU()
        self.num_layers = num_layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        max_kernel_size = max(kernel_set)
        self.idx = torch.arange(num_nodes)
        if dilated_factor>1:
            self.receptive_field = int(1+d0*(max_kernel_size-1)*(dilated_factor**num_layers-1)/(dilated_factor-1))
        else:
            self.receptive_field = d0*num_layers*(max_kernel_size-1) + 1
        # assert self.receptive_field > seq_len  - 1, f"Filter receptive field {self.receptive_field} should be  larger than sequence length {seq_len}"
        for i in range(1):
            if dilated_factor>1:
                rf_size_i = int(1 + i*(max_kernel_size-1)*(dilated_factor**num_layers-1)/(dilated_factor-1))
            else:
                rf_size_i = i*d0*num_layers*(max_kernel_size-1)+1
            new_dilation = d0
            
            for j in range(1,num_layers+1):
                if dilated_factor > 1:
                    rf_size_j = int(rf_size_i + d0*(max_kernel_size-1)*(dilated_factor**j-1)/(dilated_factor-1))
                else:
                    rf_size_j = rf_size_i+d0*j*(max_kernel_size-1)
                self.filter_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(in_channels, hidden_channels, dilation_factor=new_dilation))
                
                
                self.residual_convs.append(nn.Conv2d(in_channels=in_channels,
                                                out_channels=in_channels,
                                                kernel_size=(1, 1)))

                if self.seq_len>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=(1, self.seq_len-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))


                
                
                if self.seq_len>self.receptive_field:
                    self.norms.append(LayerNorm((hidden_channels, num_nodes, seq_len - rf_size_j + 1),elementwise_affine=True))
                else:
                    self.norms.append(LayerNorm((hidden_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=True))

                
                new_dilation *= dilated_factor
                
        # skip layer
        if self.seq_len > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, self.seq_len), bias=True)
            self.skipE = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, self.seq_len-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), bias=True)

        self.end_conv = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=out_channels if out_channels is not None else hidden_channels, 
            kernel_size=(1, 1), bias=True
        )


    def forward(self, x):
        """
        x.shape: (B, Cin, N, T)

        output.shape: (B, Cout, N, out_seq_len)
        """
        batch, _, n_nodes, seq_len = x.shape
        assert seq_len == self.seq_len, f"Sequence length {seq_len} should be {self.seq_len}"
        
        if seq_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - seq_len, 0, 0, 0))
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        for i in range(self.num_layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x # (b , c , N, p)
            s = self.skip_convs[i](s) 
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.norms[i](x,self.idx)
            
        skip = self.skipE(x) + skip
        x = F.elu(skip)
            
        x = F.elu(self.end_conv(x))
        return x









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



class HSTGAttn(nn.Module):
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
    def edge_index_extraction(self, edge_index_bi, edge_weight_bi):
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
                return edge_index_dict, edge_weight_dict
    def forward(self, x, edge_index, edge_attr):
        # x: B * (N+T) * C
        # edge_index: B,2,2*(N*T)
        # edge_attr: B*E or B * (N * T )

        for i in range(self.num_layers):
            xs = list()
            
            
            node_level_ss = list()
            node_level_tt = list()
            node_level_st = list()
            node_level_ts = list()
            semantic_level_s = list()
            semantic_level_t = list()
            for bi in range(x.shape[0]):
                x_dict = {
                    "s": x[bi][: self.node_num, :],
                    "t": x[bi][self.node_num :, :],
                }
                edge_index_bi = edge_index[bi]
                edge_weight_bi = edge_attr[bi]
                edge_index_dict,edge_weight_dict = self.edge_index_extraction(edge_index_bi, edge_weight_bi)
                
                out_dict,node_level_dict,out_att = self.convs[i](x_dict, edge_index_dict,edge_weight_dict)
                
                if self.conv_type == 'ss':
                    xi = torch.concat([out_dict["s"], x[bi][self.node_num: , :]], dim=0)
                    node_level_ss.append(node_level_dict['ss'])
                    semantic_level_s.append(out_dict['s'])
                elif self.conv_type == 'tt':
                    xi = torch.concat([x[bi][:self.node_num , :] , out_dict["t"]], dim=0)
                    node_level_tt.append(node_level_dict['tt'])
                    semantic_level_t.append(out_dict['t'])
                else:
                    xi = torch.concat([out_dict["s"], out_dict["t"]], dim=0)
                    node_level_ss.append(node_level_dict['ss'])
                    node_level_tt.append(node_level_dict['tt'])
                    node_level_st.append(node_level_dict['st'])
                    node_level_ts.append(node_level_dict['ts'])
                    semantic_level_t.append(out_dict['t'])
                    semantic_level_s.append(out_dict['s'])

                xs.append(xi)
                
                
            x = torch.stack(xs)
            nss = torch.stack(node_level_ss)
            ntt = torch.stack(node_level_tt)
            nst = torch.stack(node_level_st)
            nts = torch.stack(node_level_ts)
            smt_t = torch.stack(semantic_level_t)
            smt_s = torch.stack(semantic_level_s)

            x = F.dropout(x, p=self.dropout, training=self.training)
        return x, (nss, ntt, nst, nts), (smt_s, smt_t)






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

        # if not isinstance(in_channels, dict):
        #     in_channels = {node_type: in_channels for node_type in metadata[0]}

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
        # for node_type, in_channels in self.in_channels.items():
        #     self.proj[node_type] = Linear(in_channels, out_channels)

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
        
    ):
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict, node_level_dict = {}, {}, {}
        # Iterate over node types:
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = x.view(-1, 1, D) # .view(-1, H, D)# self.proj[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            # node_level_dict[f'{src_type}{dst_type}'] = []

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

            out = F.relu(out)
            out_dict[dst_type].append(out)
            node_level_dict[f'{src_type}{dst_type}'] = out

        # iterate over node types:
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            out, attn = group(outs, self.q[node_type], self.k_lin[node_type])
            out_dict[node_type] = out
            semantic_attn_dict[node_type] = attn

        return out_dict ,node_level_dict, semantic_attn_dict

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
