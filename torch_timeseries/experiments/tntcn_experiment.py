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
from torch_timeseries.models import TNTCN
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset


from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict


@dataclass
class TNTCNExperiment(Experiment):
    model_type: str = "TNTCN"
    
    remain_prob:float=1.0
    gcn_type:str='fagcn'
    gcn_eps:float=0.1
    casting_dim:int=32
    gcn_channel:int=32
    gc_layers:int=2
    edge_mode:int=1
    aggr_mode:str='add'
    dropout:float=0.3
    act:str='elu'
    tcn_channel:int=16
    pred_horizon:int=3
    multi_pred:bool=False
    no_time:bool=False
    no_space:bool=False
    tcn_layers:int=3
    graph_build_type: str = 'weighted_random_clip'
    output_module: str = 'tcn'
    # one_node_forecast=False
    dilated_factor:bool=2
    
    without_gc:bool = False
    without_gcn:bool = False
    
    
    n_first : bool= True
    
    def _init_model(self):
        self.model = TNTCN(
            n_nodes=self.dataset.num_features,
            input_seq_len=self.windows,
            pred_horizon=self.pred_len,
            multi_pred=False,
            graph_build_type=self.graph_build_type,
            output_module=self.output_module,
            without_gc=self.without_gc,
            no_space=self.no_space,
            no_time=self.no_time,
            act=self.act,
            gcn_type=self.gcn_type,
            gcn_eps=self.gcn_eps,
            casting_dim=self.casting_dim,
            gcn_channel=self.gcn_channel,
            gc_layers=self.gc_layers,
            edge_mode=self.edge_mode,
            aggr_mode=self.aggr_mode,
            dropout=self.dropout,
            tcn_channel=self.tcn_channel,
            tcn_layers=self.tcn_layers,
            dilated_factor=self.dilated_factor,
            without_gcn=self.without_gcn,
            n_first=self.n_first
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
        outputs = self.model(batch_x)  # torch.Size([batch_size, num_nodes])
        # single step prediction
        return outputs.reshape(batch_size, self.dataset.num_features), batch_y.reshape(batch_size, self.dataset.num_features)


def main():
    exp = TNTCNExperiment(
        dataset_type="ExchangeRate",
        data_path="./data",
        optm_type="Adam",
        batch_size=64,
        device="cuda:1",
        windows=168,
    )
    # exp.config_wandb(
    #     "project",
    #     "name"
    # )
    exp.run()



if __name__ == "__main__":
    main()
