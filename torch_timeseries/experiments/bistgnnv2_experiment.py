import random
import time
from typing import Dict, List, Type
import numpy as np
import torch
from tqdm import tqdm
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.models import BiSTGNN, BiSTGNNv2
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset


from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict


@dataclass
class BiSTGNNv2Experiment(Experiment):
    model_type: str = "BiSTGNNv2"
    
    remain_prob:float=1.0
    gcn_type:str='han'
    graph_build_type:str='adaptive'
    
    latent_dim:int=32
    gcn_layers:int=2
    tn_layers:int=1
    
    heads:int = 1
    negative_slope:float = 0.2
    
    node_static_embed_dim :int = 16
    dropout:float=0.3
    act:str='elu'
    tcn_channel:int=16
    rebuild_time:bool=True
    rebuild_space:bool=True
    tcn_layers:int=3
    dilated_factor:int = 2
    
    
    without_tn_module:bool = False
    without_gcn:bool = False
    
    def _init_model(self):
        self.model = BiSTGNNv2(
            seq_len=self.windows,
            num_nodes=self.dataset.num_features,
            temporal_embed_dim=4, # 4 for hour embedding
            latent_dim=self.latent_dim,
            
            heads=self.heads,
            negative_slope=self.negative_slope,

            gcn_layers=self.gcn_layers,
            dropout=self.dropout,
            graph_build_type=self.graph_build_type,
            graph_conv_type=self.gcn_type,
            tn_layers=self.tn_layers,
            rebuild_time=self.rebuild_time,
            rebuild_space=self.rebuild_space,
            node_static_embed_dim=self.node_static_embed_dim,
            tcn_layers=self.tcn_layers,
            dilated_factor=self.dilated_factor,
            tcn_channel=self.tcn_channel,
            act=self.act
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
        return outputs.reshape(batch_size, self.dataset.num_features), batch_y.reshape(batch_size, self.dataset.num_features)


def main():
    exp = BiSTGNNv2Experiment(
        dataset_type="ExchangeRate",
        data_path="./data",
        optm_type="Adam",
        batch_size=64,
        device="cuda:0",
        windows=168,
    )
    # exp.config_wandb(
    #     "project",
    #     "name"
    # )
    exp.run()



if __name__ == "__main__":
    main()
