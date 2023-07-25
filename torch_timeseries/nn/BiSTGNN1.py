import copy

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from class_resolver.contrib.torch import activation_resolver
from torch_geometric.nn import FAConv


class BiSTGNN(nn.Module):
    def __init__(self, input_seq_len, n_nodes, node_dim=512):
        super().__init__()

        self.static_nodes_embeddings = nn.parameter.Parameter(
            torch.randn(n_nodes, node_dim)
        )  #  nn.Embedding(n_nodes , embedding_dim=static_node_embeding_dim) #
        self.spatial_att_embe_layer = SpatialAttentionLayer(
            in_dim=input_seq_len, node_dim=node_dim
        )

        # temporal encoding layers
        self.gru = nn.GRU(n_nodes, hidden_size=node_dim, batch_first=True)
        self.temporal_projection_layer = nn.Linear(n_nodes, node_dim)

    def forward(self, x):
        # input :  ( B, N, L)
        # N : node dim , L : sequence dim

        dynamic_node_embedding = self.spatial_att_embe_layer(
            x, self.static_nodes_embeddings
        )  # ( B, N, static_node_embeding_dim)

        # temporal embedding

        output, _ = self.gru(x.transpose(1, 2))  # ( B , L ,N )
        temporal_encoding = self.temporal_projection_layer(output)  # ( B, L, node_dim)

        act = nn.Sigmoid()
        pre_adj = torch.einsum(
            "bnd,bld->bln", dynamic_node_embedding, temporal_encoding
        )
        adj = act(pre_adj)

        pass


class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_dim, n_heads=8, attention_dropout: float = 0.1, node_dim=512):
        """Full attention of queries and keys, the vallina version of attention in Paper :
        A. Vaswani et al., “Attention is All you Need”.
        """
        super(FullAttention, self).__init__()

        self.k_projectoin_layer = nn.Linear(in_dim, node_dim)
        self.v_projectoin_layer = nn.Linear(in_dim, node_dim)

        self.attention = FullAttention(attention_dropout)

    def forward(self, x, static_node_embeddings):
        # x : ( B, N ,L)
        # static_node_embeddings :  (N, node_dim)
        # output: (B, N, node_dim)
        k = self.k_projectoin_layer(x)
        v = self.v_projectoin_layer(x)

        V, A = self.attention(static_node_embeddings, k, v)

        return V, A



class FullAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1):
        """Full attention of queries and keys, the vallina version of attention in Paper :
        A. Vaswani et al., “Attention is All you Need”.

        Args:
            mask_flag (bool, optional): whether to apply the masked attention
            scale (_type_, optional): _description_. Defaults to None.
            attention_dropout (float, optional): _description_. Defaults to 0.1.
        """
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """
        B : batch size
        L,S : input sequence length , this is the timeseries dimension
        H : num of heads
        E,D : dimention of input tokens, defaults to 512

        Args:
            queries (torch.Tensor): queries with shape (B, L, H, E)
            keys (torch.Tensor): keys with shape (B, S, H ,D)
            values (torch.Tensor): values with shape (B, S, H ,D)
            attn_mask (torch.Tensor, optional): attention matrix. Defaults to None.

        Returns:
            (attention_outputs, attention_scores): output attention outputs (shape: (B, L, H, E)) and attention scores
        """

        # get the scores attention matrix
        N, D = queries.shape
        B, N, D = values.shape

        # B, L, H, E = queries.shape
        # _, S, _, D = values.shape
        scale = 1.0 / torch.sqrt(D)
        scores = torch.einsum("nd,bnd->bnn", queries, keys)

        # attent the values matrix
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bns,bnd->bsd", A, values)

        return (V.contiguous(), A)
