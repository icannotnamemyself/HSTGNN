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
from torch_timeseries.layers.graphsage import MyGraphSage, MyFAGCN


class STI(nn.Module):
    def __init__(
        self,
        seq_len,
        num_nodes,
        temporal_embed_dim,
        graph_build_type="attsim_direc_tt_mask", # "predefined_adaptive"
        graph_conv_type="weighted_han_update3",
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
        super(STI, self).__init__()