#   cl_decay_steps: 2000
#   filter_type: dual_random_walk
#   horizon: 12
#   input_dim: 2
#   l1_decay: 0
#   max_diffusion_step: 2
#   num_nodes: 325
#   num_rnn_layers: 1
#   output_dim: 1
#   rnn_units: 128
#   seq_len: 12
#   use_curriculum_learning: true
#   dim_fc: 583408


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
import torch_timeseries.models.GTS as GTS
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
class GTSxperiment(Experiment):
    model_type: str = "GTS"

    
    temperature : float = 0.5 
    # cl_decay_steps: 2000
    filter_type : str =  'dual_random_walk'
    # horizon: 12
    input_dim: int = 1
    # l1_decay: 0
    max_diffusion_step : int = 2
    # num_nodes: 325
    num_rnn_layers:int = 1
    output_dim:int = 1
    rnn_units: int = 128
    seq_len: int =12
    use_curriculum_learning: bool = True
    dim_fc: int =583408
     

    def _init_model(self):
        GTS.device = self.device
        
        node_feature = torch.rand(self.dataset.num_features)
        
        model_kwargs={
              "filter_type" : "dual_random_walk",
              "horizon": self.seq_len,
              "input_dim": self.input_dim,
            #   "l1_decay": self.l1,
              "max_diffusion_step": self.max_diffusion_step,
              "num_nodes": self.dataset.num_features,
              "num_rnn_layers": self.num_rnn_layers,
              "output_dim": self.output_dim,
              "rnn_units": self.rnn_units,
              "seq_len": self.seq_len,
              "use_curriculum_learning": self.use_curriculum_learning,
              "dim_fc": self.dim_fc,
        }
        self.model = GTS.GTSModel(self.temperature, None , **model_kwargs)
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        
        batch_x = batch_x.permute(1, 0, 2) #  (seq_len, batch_size, num_sensor * input_dim)
        outputs = self.model(batch_x)  # (horizon, batch_size, num_sensor * output)
        outputs = outputs.permute(1, 0, 2)
        return outputs, batch_y




def main():
    exp = GTSxperiment(
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
