from typing import List, Tuple
import pyro
import torch
from torch import nn
from torch_timeseries.layers.HAN import HAN
from torch_timeseries.layers.HGT import HGT
from torch_timeseries.layers.MAGNN import MAGNN
from torch_timeseries.layers.dilated_convolution import DilatedConvolution
# from torch_timeseries.layers.weighted_han_update9 import WeightedHAN as WeightedHANUpdate9
from torch_timeseries.layers.HSTGAttn import HSTGAttn
from torch_timeseries.layers.GraphSage import GraphSage, GCN


class HSTGNN(nn.Module):
    def __init__(
        self,
        seq_len,
        num_nodes,
        temporal_embed_dim,
        graph_build_type="attsim_direc_tt_mask3", # "predefined_adaptive"
        graph_conv_type="HSTGAttn",
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
            self.tn_modules.append(
                HSTG(
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
            
            
        self.output_layer = DilatedConvolution(
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


class HSTG(nn.Module):
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
        self.graph_constructor = GraphConstructor(predefined_adj=None,latent_dim=latent_dim,tt_mask=True, N=num_nodes, O=seq_len)


        if not self.without_gcn:
            if graph_conv_type == 'GraphSage':
                self.graph_conv = GraphSage(latent_dim, latent_dim, gcn_layers,act='elu')
            elif graph_conv_type == 'GCN':
                self.graph_conv = GCN(latent_dim, latent_dim, gcn_layers,act='elu')
            elif graph_conv_type == 'HAN':
                self.graph_conv = HAN(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act
                )
            elif graph_conv_type == 'HGT':
                self.graph_conv = HGT(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act
                )
            elif graph_conv_type == 'HSTGAttn':
                self.graph_conv = HSTGAttn(
                    num_nodes, seq_len, latent_dim, latent_dim,latent_dim, gcn_layers,
                    heads=self.heads, negative_slope=self.negative_slope, dropout=self.dropout,act=self.act,
                    conv_type=self.conv_type
                )
            elif graph_conv_type == 'MAGNN':
                self.graph_conv = MAGNN(
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
        B, N, T = x.size()
        assert N == self.num_nodes
        # set dtype to LongType
        static_embeding = self.static_node_embedding.weight.expand(B, N, -1)

        x = torch.cat((x, static_embeding), dim=2)

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
        B, T, N = x.size()
        assert T == self.seq_len
        static_embeding = self.static_node_embedding.weight.expand(B, T, -1)
        if self.temporal_embed_dim  > 0:
            x = torch.concat((x, x_enc_mark, static_embeding), dim=2)
        else:
            x = torch.concat((x, static_embeding), dim=2)

        temporal_encode , _= self.temporal_projection(x)
        return temporal_encode




class GraphConstructor(nn.Module):
    def __init__(self, predefined_adj=None, latent_dim=32,tt_mask=True, alpha=3, N=0, O=0):
        super(GraphConstructor, self).__init__()
        self.predefined_adj =predefined_adj
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
        x1_expanded = x1.unsqueeze(2)  # 维度变为 [B, N1, 1, D]
        x2_expanded = x2.unsqueeze(1)  # 维度变为 [B, 1, N2, D]

        x1_tiled = x1_expanded.repeat(1, 1, N2, 1)  # [B, N1, N2, D]
        x2_tiled = x2_expanded.repeat(1, N1, 1, 1)  # [B, N1, N2, D]
        x_combined = torch.cat((x1_tiled, x2_tiled), dim=-1)  # [B, N, O, 2D]
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
        return adj 

    def build_ss_graph(self, x1, x2):
        B, N1, D = x1.size()
        B ,N2, D = x2.size()
        
        nodevec1 = torch.tanh(self.alpha*self.ss_x1_lin(x1))
        nodevec2 = torch.tanh(self.alpha*self.ss_x2_lin(x2))
        adj =  self.alpha*(torch.einsum("bnf, bmf -> bnm", nodevec1, nodevec2)  -   torch.einsum("bnf, bmf -> bnm", nodevec2, nodevec1))
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
