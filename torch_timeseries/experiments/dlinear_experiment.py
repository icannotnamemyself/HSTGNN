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
from torch_timeseries.models import DLinear
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset


from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict


@dataclass
class DLinearExperiment(Experiment):
    model_type: str = "DLinear"

    individual : bool = False

    def _init_model(self):
        self.model = DLinear(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        outputs = self.model(batch_x)  # torch.Size([batch_size, output_length, num_nodes])
        return outputs.squeeze(1), batch_y.squeeze(1)


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
