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
from torch_timeseries.layers.tcn_output9_norm import TCNOuputLayer as TCNOuputLayer9
from torch_timeseries.layers.nlinear_output import NlinearOuputLayer
from torch_timeseries.layers.weighted_han import WeightedHAN
from torch_timeseries.layers.weighted_han_update import WeightedHAN as WeightedHANUpdate
from torch_timeseries.layers.weighted_han_update2 import WeightedHAN as WeightedHANUpdate2
from torch_timeseries.layers.graphsage import MyGraphSage, MyFAGCN
from torch_timeseries.nn.dil_encoder import LayerNorm


class HSTGNN(nn.Module):
    def __init__(
        self,
        seq_len,
        num_nodes,
        graph_build_type="adaptive", # "predefined_adaptive"
        graph_conv_type="weighted_han",
        # output_layer_type='tcn8',
        heads=1,
        negative_slope=0.2,
        gcn_layers=2,
        rebuild_time=True,
        rebuild_space=True,
        node_static_embed_dim=16,
        latent_dim=16,
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
        in_dim=1,
        normalization=True,
        conv_type='all',
        num_layers=4,
    ):
        super(HSTGNN, self).__init__()

        self.num_nodes = num_nodes
        self.static_embed_dim = node_static_embed_dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.out_seq_len = out_seq_len
        self.seq_len = seq_len
        self.self_loop_eps = self_loop_eps
        self.without_tn_module = without_tn_module
        self.kernel_set = kernel_set
        self.normalization =normalization
        self.in_dim = in_dim
        self.conv_type = conv_type
        self.tcn_channel =tcn_channel
        self.graph_build_type  = graph_build_type
        self.graph_conv_type  = graph_conv_type
        self.gcn_layers :int = gcn_layers
        self.num_layers :int = num_layers
        self.dilated_factor= dilated_factor
        self.act = act
        self.without_tn_module = without_tn_module
        self.without_gcn = without_gcn
        self.d0 = d0
        self.dropout = dropout
        self.start_conv1 = nn.Conv2d(in_dim, 2, (1, 1))  # to ( s and t)
        self.start_conv2 = nn.Conv2d(2, tcn_channel, (1, 1))
        
        self.predefined_adj = predefined_adj

        self.rebuild_time = rebuild_time
        self.rebuild_space = rebuild_space

        if not self.without_tn_module:
            self.tn_modules = nn.ModuleList()
                
        self._build_tcns()

    def _build_tcns(self):
        
        # d0 = self.d0
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        
        # num_layers= self.num_layers
        max_kernel_size = max(self.kernel_set)
        self.idx = torch.arange(self.num_nodes)
        dilated_factor = self.dilated_factor
        if dilated_factor>1:
            self.receptive_field = int(1+self.d0*(max_kernel_size-1)*(dilated_factor**self.num_layers-1)/(dilated_factor-1))
        else:
            self.receptive_field = self.d0*self.num_layers*(max_kernel_size-1) + 1
        
        
        
        L0 = max(self.seq_len , self.receptive_field)
        
        # assert self.receptive_field > seq_len  - 1, f"Filter receptive field {self.receptive_field} should be  larger than sequence length {seq_len}"
        for i in range(1):
            if dilated_factor>1:
                rf_size_i = int(1 + i*(max_kernel_size-1)* self.d0 *(dilated_factor**self.num_layers-1)/(dilated_factor-1))
            else:
                rf_size_i = i*self.d0*self.num_layers*(max_kernel_size-1)+1
            new_dilation = self.d0
            for j in range(1,self.num_layers+1):
                # input seq len
                
                
                if dilated_factor > 1:
                    rf_size_j = int(rf_size_i + self.d0*(max_kernel_size-1)*(dilated_factor**j-1)/(dilated_factor-1))
                else:
                    rf_size_j = rf_size_i+self.d0*j*(max_kernel_size-1)
                
                rj = (self.d0 * (max_kernel_size - 1) * (self.dilated_factor**j - 1))/ (self.dilated_factor - 1) + j
                Lj =  int(L0 + j - rj)
                
                self.filter_convs.append(DilatedInception(self.tcn_channel, self.tcn_channel, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(self.tcn_channel, self.tcn_channel, dilation_factor=new_dilation))
                
                
                self.residual_convs.append(nn.Conv2d(in_channels=self.tcn_channel,
                                                out_channels=self.tcn_channel,
                                                kernel_size=(1, 1)))

                if self.seq_len>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.tcn_channel,
                                                    out_channels=self.tcn_channel,
                                                    kernel_size=(1, self.seq_len-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.tcn_channel,
                                                    out_channels=self.tcn_channel,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                
                if self.seq_len>self.receptive_field:
                    self.norms.append(LayerNorm((self.tcn_channel, self.num_nodes, self.seq_len - rf_size_j + 1),elementwise_affine=True))
                else:
                    self.norms.append(LayerNorm((self.tcn_channel, self.num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=True))

                
                new_dilation *= dilated_factor




                # STModule
                if not self.without_tn_module:
                    self.tn_modules.append(
                        TNModule(
                            self.num_nodes,
                            self.seq_len,
                            self.latent_dim,
                            self.gcn_layers,
                            graph_build_type=self.graph_build_type,
                            graph_conv_type=self.graph_conv_type,
                            conv_type=self.conv_type,
                            heads=self.heads,
                            negative_slope=self.negative_slope,
                            dropout=self.dropout,
                            act=self.act,
                            predefined_adj=self.predefined_adj,
                            self_loop_eps=self.self_loop_eps,
                            without_gcn=self.without_gcn,
                            s_input_size=int(Lj*int(self.tcn_channel/2)),
                            t_input_size=self.num_nodes*int(self.tcn_channel/2)
                        )
                    )


                
        # skip layer
        if self.seq_len > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.tcn_channel, kernel_size=(1, self.seq_len), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.tcn_channel, out_channels=self.tcn_channel, kernel_size=(1, self.seq_len-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.tcn_channel, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.tcn_channel, out_channels=self.tcn_channel, kernel_size=(1, 1), bias=True)

        self.end_conv = nn.Conv2d(
            in_channels=self.tcn_channel, 
            out_channels=self.out_seq_len, 
            kernel_size=(1, 1), bias=True
        )

    def forward(self, x, x_enc_mark):
        """
        in :  (B, N, T)
        out:  (B, N, latent_dim)
        """
        B, N, T = x.size()
        if self.normalization:
            seq_last = x[:,:,-1:].detach()
            x = x - seq_last

        s_channel = int(self.tcn_channel/2)
        t_channel = int(self.tcn_channel/2)
        
        x = x.unsqueeze(1) # (B, 1 ,N, T)
        batch, _, n_nodes, seq_len = x.shape
        assert seq_len == self.seq_len, f"Sequence length {seq_len} should be {self.seq_len}"
        
        if seq_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - seq_len, 0, 0, 0))
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        x = self.start_conv1(x) # (B, 2 ,N, T)
        x = self.start_conv2(x) # (B, tcn_channel ,N, T)
        for i in range(self.num_layers):
            
            # TCN
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
            
            
            # GCN
            if not self.without_tn_module:
                B, C, N, T = x.size()
                x1 = x.clone()
                s_size = s_channel * T 
                t_size = t_channel * N 
                xs = x1[:, :s_channel, :, :].permute(0, 2, 1, 3).contiguous().view(B, N, s_size) # B, N, s_channel,T -> (B, N, s_size)
                xt = x1[:, s_channel:, :, :].permute(0, 3, 1, 2).contiguous().view(B, T, t_size) #  ->(B, T, t_channel,N)  -> (B, T, t_size) 
                xs, xt = self.tn_modules[i](xs, xt)  # (B, N, s_size) (B, T, t_size)
                x1[:, :s_channel, :, :] = xs.view(B, N, s_channel, T).permute(0, 2, 1, 3)
                x1[:, s_channel:, :, :] = xt.view(B, T, t_channel, N).permute(0, 2, 3, 1)
                
                x = x1
            

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.norms[i](x,self.idx)
            
            
            

                
        skip = self.skipE(x) + skip
        x = F.elu(skip)
            
        x = F.elu(self.end_conv(x)).squeeze(3) # (B, O, N)
        
        if self.normalization:
            x = (x.transpose(1,2) + seq_last).transpose(1,2)
        return x
            
        
        # outputs = list()
        # if self.rebuild_space:
        #     n_output = self.feature_rebuild(Xs)  # (B, N, T)
        #     outputs.append(n_output.unsqueeze(1))
        # if self.rebuild_time:
        #     t_output,_ = self.time_rebuild(Xt)  # (B, T, N)
        #     outputs.append(t_output.unsqueeze(1).transpose(2, 3))
        # outputs.append(x.unsqueeze(1))  # （B, 1/2/3, N, T)

        # output module
        X = torch.cat(outputs, dim=1)  # （B, 1/2/3, N, T)
        X = self.output_layer(X)  # (B, O, N)
        
        
        if self.normalization:
            X = (X.transpose(1,2) + seq_last).transpose(1,2)

        return X


class TNModule(nn.Module):
    def __init__(
        self, num_nodes, seq_len, latent_dim, gcn_layers, s_input_size, t_input_size, graph_build_type="adaptive",
        graph_conv_type='fastgcn5',conv_type='all',heads=1,negative_slope=0.2,dropout=0.0,act='elu',self_loop_eps=0.5,without_gcn=False,
        predefined_adj=None, node_static_embed_dim=16,
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
            self.graph_constructor = MLPSimConstructor(predefined_adj=predefined_adj,latent_dim=latent_dim,self_loop_eps=self.self_loop_eps)
            print("graph_build_type dis mlpsim")


        self.spatial_encoder = SpatialEncoder(
            seq_len,
            num_nodes,
            input_dim=s_input_size,
            static_embed_dim=node_static_embed_dim,
            latent_dim=latent_dim,
        )
        self.temporal_encoder = TemporalEncoder(
            seq_len,
            num_nodes,
            input_dim=t_input_size,
            static_embed_dim=node_static_embed_dim,
            latent_dim=latent_dim,
        )


        self.feature_rebuild = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                torch.nn.ELU(),
                nn.Linear(latent_dim, s_input_size),
            )

        self.time_rebuild = nn.GRU(latent_dim, t_input_size, batch_first=True)

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

    def forward(self, xs, xt):
        """
        Args:
            X :  (B, N+T, latent_dim)

        Returns:
            X: (B, N+T, latent_dim)
        """
        
        
        
        Xs = self.spatial_encoder(xs)
        Xt = self.temporal_encoder(xt)
        X = torch.concat([Xs, Xt], dim=1)  # (B, N+T, latent_dim)

        
        # Xs = X[:, : self.num_nodes, :]
        # Xt = X[:, self.num_nodes :, :]

        batch_adj, batch_indices, batch_values = self.graph_constructor(
            Xs, Xt
        )  # spatial and temporal adjecent matrix
        if not self.without_gcn:
            X = self.graph_conv(X, batch_indices, batch_values)
        
        Xs = self.feature_rebuild(X[:, : self.num_nodes, :])
        Xt, _ = self.time_rebuild(X[:, self.num_nodes: , :])
        return Xs, Xt


class SpatialEncoder(nn.Module):
    def __init__(self, seq_len, num_nodes,input_dim, static_embed_dim=32, latent_dim=256):
        super(SpatialEncoder, self).__init__()

        self.num_nodes = num_nodes
        self.static_embed_dim = static_embed_dim
        self.latent_dim = latent_dim

        self.static_node_embedding = nn.Embedding(num_nodes, static_embed_dim)
        input_dim = input_dim + static_embed_dim
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
        self, seq_len, num_nodes,input_dim, static_embed_dim=32, latent_dim=64
    ):
        super(TemporalEncoder, self).__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.static_embed_dim = static_embed_dim
        self.static_node_embedding = nn.Embedding(seq_len, static_embed_dim)
        self.temporal_projection = nn.GRU(
            input_dim,latent_dim, batch_first=True
        )

    def forward(self, x):
        """
        x :  (B, T, N)
        x_enc_mark : (B, T, D_t)


        out:  (B, T, latent_dim)
        """
        # 获取输入张量x的形状
        B, T, D = x.size()
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



class TCNOuputLayer(nn.Module):
    def __init__(self,input_seq_len,out_seq_len,tcn_layers,dilated_factor,in_channel,tcn_channel) -> None:
        super().__init__()
        self.channel_layer = nn.Conv2d(in_channel, tcn_channel, (1, 1))
        self.tcn =TCN(
                    input_seq_len, tcn_channel, tcn_channel,
                    out_seq_len=out_seq_len, num_layers=tcn_layers,
                    dilated_factor=dilated_factor
                )
        self.end_layer = nn.Conv2d(tcn_channel, 1, (1, 1))
        self.act = nn.ELU()
        
    def forward(self, x):
        output = self.channel_layer(x)
        output = self.act(self.tcn(output))  # (B, C, N, out_len)
        output = self.end_layer(output).squeeze(1).transpose(1, 2)  # (B, out_len, N)
        return output





import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.dilated_inception import DilatedInception

class TCN(nn.Module):
    def __init__(
        self, seq_len, in_channels, hidden_channels, out_seq_len=1, out_channels=None, num_layers=2, 
        dilated_factor=2, dropout=0
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dropout = dropout
        self.act = nn.ReLU()
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

