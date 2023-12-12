import random
import time
from typing import Dict, List, Type
import numpy as np
import torch
from tqdm import tqdm
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset, TimeSeriesStaticGraphDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.models import  ESG
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

import argparse
import numpy as np
import os
import pandas as pd

import os


from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict, field


@dataclass
class ESGExperiment(Experiment):
    model_type: str = "ESG"
    
    
    
    dy_embedding_dim : int = 20
    dy_interval:  List[int] = field(default_factory=lambda:[31,31,21,14,1])
    # num_nodes: int,
    # seq_length: int,
    # pred_len : int,
    # in_dim: int,
    # out_dim: int,
    n_blocks: int = 1
    n_layers: int = 5           
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int  = 64
    kernel_set : List[int] = field(default_factory=lambda:[2,3,6,7])
    dilation_exp : int  = 2
    gcn_depth: int = 2                           
    # (node_fea.shape[0]-18)*16 fc_dim: int,
    st_embedding_dim : int =40
    dropout : float =0.3
    propalpha : float =0.05
    layer_norm_affline: bool =True            
     

    def _init_model(self):
        
        node_fea = self.dataloader.train_dataset.dataset.data
        node_fea = torch.tensor(node_fea).to(self.device).float()
        self.model = ESG(
                 dy_embedding_dim = self.dy_embedding_dim,
                 dy_interval = self.dy_interval,
                 num_nodes = self.dataset.num_features,
                 seq_length = self.windows,
                 pred_len =self.pred_len,
                 in_dim=1,
                 out_dim=1,
                 n_blocks=self.n_blocks,
                 n_layers=self.n_layers,                
                 conv_channels=self.conv_channels,
                 residual_channels=self.residual_channels,
                 skip_channels=self.skip_channels,
                 end_channels=self.end_channels,
                 kernel_set=self.kernel_set,
                 dilation_exp=self.dilation_exp,
                 gcn_depth=self.gcn_depth,                             
                 device=self.device,
                 fc_dim=(node_fea.shape[0]-18)*16,
                 st_embedding_dim=self.st_embedding_dim,
                 static_feat=node_fea,
                 dropout=self.dropout,
                 propalpha=self.propalpha,
                 layer_norm_affline=self.layer_norm_affline            
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
        batch_x = batch_x.transpose(1,2).unsqueeze(1)
        outputs = self.model(batch_x)  # torch.Size([batch_size, O, num_nodes, out_dim])
        outputs = outputs.squeeze(-1)
        return outputs, batch_y




def main():
    exp = ESGExperiment(
        dataset_type="DummyDatasetGraph",
        data_path="./data",
        optm_type="Adam",
        horizon=1,
        pred_len=1,
        batch_size=12,
        device="cuda:0",
        windows=12,
    )
    # exp.config_wandb(
    #     "project",
    #     "name"
    # )
    exp.run()



if __name__ == "__main__":
    main()
