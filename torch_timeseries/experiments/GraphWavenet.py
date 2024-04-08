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
from torch_timeseries.models.GraphWavenet import gwnet as GraphWaveNet

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.utils.adj import adj_to_edge_index_weight


@dataclass
class GraphWavenetExperiment(Experiment):
    model_type: str = "GraphWavenet"
    layers : int = 2
        
    def _init_model(self):
        assert isinstance(self.dataset, TimeSeriesStaticGraphDataset), "dataset must be of type TimeSeriesStaticGraphDataset"
        # adj = torch.tensor(self.dataset.adj).to(self.device, dtype=torch.float32)
        
        
        adj = torch.tensor(self.dataset.adj).to(self.device).float()
        self.model = GraphWaveNet(
            device=self.device,
            num_nodes=self.dataset.num_features,
            gcn_bool=True,
            addaptadj=True,
            aptinit=adj,
            in_dim=self.windows,
            out_dim=self.pred_len,
            layers=self.layers,
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
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device)
        batch_y_date_enc = batch_y_date_enc.to(self.device)
        input_x = batch_x.unsqueeze(1)
        input_x = input_x.transpose(
            2, 3
        )  # torch.Size([batch_size, 1, num_nodes, windows])
        
        # B C N T
        outputs = self.model(input_x.transpose(1,3))  # torch.Size([batch_size, seq_len, num_nodes, 1])
        outputs = outputs.squeeze(3) # ( B , O , N)
        if self.pred_len == 1:
            return outputs.squeeze(1), batch_y.squeeze(1)
        elif self.pred_len > 1:
            return outputs, batch_y
        return outputs, batch_y


def main():
    exp = GraphWavenetExperiment(
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




def cli():
    import fire
    fire.Fire(GraphWavenetExperiment)

if __name__ == "__main__":
    cli()

