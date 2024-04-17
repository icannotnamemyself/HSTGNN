from typing import List
import torch
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset, TimeSeriesStaticGraphDataset
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.models import  HSTGNN



import wandb

from dataclasses import dataclass, asdict, field


@dataclass
class HSTGNNExperiment(Experiment):
    model_type: str = "HSTGNN"
    gcn_type : str = 'HSTGAttn'
    # graph_build_type:str='attsim_direc_tt_mask3_no_predefined'
    graph_build_type:str='attsim_direc_tt_mask3_no_predefined'
    output_layer_type:str='tcn8'
    conv_type:str='all'
    
    latent_dim:int=16
    gcn_layers:int=2
    tn_layers:int=1
    
    heads:int = 1
    negative_slope:float = 0.2
    
    node_static_embed_dim :int = 16
    dropout:float=0.0
    act:str='elu'
    tcn_channel:int=16
    rebuild_time:bool=True
    rebuild_space:bool=True
    tcn_layers:int = 5
    dilated_factor:int = 2
    without_tn_module:bool = False
    without_gcn:bool = False
    d0 : int = 1
    kernel_set : List[int] = field(default_factory=lambda:[2,3,6,7])
    normalization : bool = True

    def _init_model(self):
        predefined_NN_adj = None
        padded_A = None
        if isinstance(self.dataset, TimeSeriesStaticGraphDataset) and self.pred_len > 1:
            predefined_NN_adj = torch.tensor(self.dataset.adj).to(self.device)
            D = torch.diag(torch.sum(predefined_NN_adj, dim=1))
            D_sqrt_inv = torch.sqrt(torch.inverse(D))
            normalized_predefined_adj = D_sqrt_inv @predefined_NN_adj @ D_sqrt_inv
            padded_A = torch.nn.functional.pad(normalized_predefined_adj, (0, self.windows, 0, self.windows), mode='constant', value=0).float()

        else:
            padded_A = None
        self.padded_A = padded_A
        if isinstance(self.dataset, PeMS_D7):
            temporal_embed_dim = 0
        else:
            temporal_embed_dim = 4
        print("self gcn type", self.conv_type)
        self.model = HSTGNN(
            normalization=self.normalization,
            seq_len=self.windows,
            num_nodes=self.dataset.num_features,
            temporal_embed_dim=temporal_embed_dim, # 4 for hour embedding
            latent_dim=self.latent_dim,
            predefined_adj=padded_A,
            heads=self.heads,
            negative_slope=self.negative_slope,
            gcn_layers=self.gcn_layers,
            dropout=self.dropout,
            graph_build_type=self.graph_build_type,
            graph_conv_type=self.gcn_type,
            rebuild_time=self.rebuild_time,
            rebuild_space=self.rebuild_space,
            out_seq_len=self.pred_len,
            node_static_embed_dim=self.node_static_embed_dim,
            tcn_layers=self.tcn_layers,
            dilated_factor=self.dilated_factor,
            tcn_channel=self.tcn_channel,
            act=self.act,
            output_layer_type=self.output_layer_type,
            without_tn_module=self.without_tn_module,
            without_gcn=self.without_gcn,
            conv_type=self.conv_type,
            d0=self.d0
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        batch_x = batch_x.transpose(1,2)
        outputs = self.model(batch_x,batch_x_date_enc)  # torch.Size([batch_size, num_nodes])
        # single step prediction
        return outputs, batch_y


def main():
    exp = HSTGNNExperiment(
        dataset_type="DummyDatasetGraph",
        data_path="./data",
        optm_type="Adam",
        horizon=1,
        graph_build_type='predefined_adaptive',
        pred_len=3,
        batch_size=12,
        device="cuda:0",
        windows=12,
    )
    # exp.config_wandb(
    #     "project",
    #     "name"
    # )
    exp.run()

def cli():
    import fire
    fire.Fire(HSTGNNExperiment)

if __name__ == "__main__":
    cli()
