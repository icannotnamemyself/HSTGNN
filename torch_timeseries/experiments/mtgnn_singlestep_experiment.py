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
from torch_timeseries.nn.mtgnn import MTGNN
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset


from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict


@dataclass
class MTGNNExperiment(Experiment):
    model_type: str = "MTGNN"
    gcn_true: bool = True
    gcn_depth: int = 2
    dropout: float = 0.3
    subgraph_size: int = 3
    node_dim: int = 40
    dilation_exponential: float = 2
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    in_dim: int = 1
    # out_dim:int=12
    layers: int = 5
    propalpha: float = 0.05
    tanhalpha: float = 3
    layer_norm_affline: bool = False
    skip_layer: bool = True
    residual_layer: bool = True

    invtrans_loss: bool = False

    def _init_model(self):
        assert (
            self.subgraph_size <= self.dataset.num_features
        ), f"graph size {self.subgraph_size} have to be small than data columns :{self.dataset.num_features}"
        self.model = MTGNN(
            gcn_true=self.gcn_true,
            buildA_true=True,
            gcn_depth=self.gcn_depth,
            num_nodes=self.dataset.num_features,
            device=self.device,
            predefined_A=None,
            static_feat=None,
            dropout=self.dropout,
            subgraph_size=self.subgraph_size,
            node_dim=self.node_dim,
            dilation_exponential=self.dilation_exponential,
            conv_channels=self.conv_channels,
            residual_channels=self.residual_channels,
            skip_channels=self.skip_channels,
            end_channels=self.end_channels,
            seq_length=self.windows,
            in_dim=self.in_dim,
            out_dim=self.pred_len,
            layers=self.layers,
            propalpha=self.propalpha,
            tanhalpha=self.tanhalpha,
            layer_norm_affline=self.layer_norm_affline,
            skip_layer=self.skip_layer,
            residual_layer=self.residual_layer,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
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
        outputs = self.model(input_x)  # torch.Size([batch_size, seq_len, num_nodes, 1])
        outputs = outputs.squeeze(3)
        if self.invtrans_loss:
            pred = self.scaler.inverse_transform(outputs)
            batch_y = self.scaler.inverse_transform(batch_y)
        else:
            pred = outputs
            batch_y = batch_y
        return pred.squeeze(1), batch_y.squeeze(1)


# def main():
#     exp = MTGNNExperiment(
#         dataset_type="ExchangeRate",
#         data_path="./data",
#         optm_type="Adam",
#         batch_size=64,
#         device="cuda:3",
#         windows=168,
#         epochs=1,
#         lr=0.0003,
#         horizon=3,
#         residual_channels=16,
#         gcn_true=False,
#         l2_weight_decay=0.0005,
#         skip_channels=16,
#         residual_layer=16,
#         layers=5,
#         pred_len=1,
#         subgraph_size=3,
#         end_channels=16,
#         seed=42,
#         scaler_type="StandarScaler",
#         wandb=False,
#     )
#     exp.config_wandb(
#         "project",
#         "name"
#     )
#     exp.runs()


# if __name__ == "__main__":
#     main()
