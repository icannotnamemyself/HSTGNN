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
from torch_timeseries.models import BiSTGNN, BiSTGNNv2
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
from torch_geometric_temporal.nn.attention import STConv, ASTGCN

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.utils.adj import adj_to_edge_index_weight


@dataclass
class ASTGCNExperiment(Experiment):
    model_type: str = "ASTGCN"
    K : int = 2
    nb_block : int = 1
    nb_chev_filter : int = 1
    nb_time_filter : int = 1
    time_strides : int = 2
    
    hidden_channels : int = 16
    kernel_size :int = 2
    def _init_model(self):
        assert isinstance(self.dataset, TimeSeriesStaticGraphDataset), "dataset must be of type TimeSeriesStaticGraphDataset"
        self.model = ASTGCN(
            nb_block=self.nb_block,
            in_channels=1,
            K=self.K,
            nb_chev_filter=self.nb_chev_filter,
            nb_time_filter=self.nb_time_filter,
            time_strides=self.time_strides,
            len_input=self.windows,
            num_for_predict=self.pred_len,
            num_of_vertices=self.dataset.num_features,
        )
        
        self.model = self.model.to(self.device)

        self.edge_index, self.edge_weight = adj_to_edge_index_weight(self.dataset.adj)
        self.edge_index = torch.tensor(self.edge_index).to(self.device)
        self.edge_weight = torch.tensor(self.edge_weight).to(self.device)

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
        batch_x = batch_x.unsqueeze(-1)  # (B, T, N, D)
        batch_x = batch_x.transpose(1,2).transpose(2,3)  # (B, N, D, T)
        outputs = self.model(batch_x, self.edge_index) # (B, N, O)
        outputs = outputs.transpose(1,2)  # (B, O, N)
        return outputs, batch_y


def main():
    exp = ASTGCNExperiment(
        dataset_type="DummyDatasetGraph",
        data_path="./data",
        optm_type="Adam",
        batch_size=64,
        device="cuda:0",
        pred_len=3,
        horizon=1,
        windows=16,
    )
    # exp.config_wandb(
    #     "project",
    #     "name"
    # )
    exp.run()



if __name__ == "__main__":
    main()
