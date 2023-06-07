import random
import time
from typing import Dict, List, Type
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.data.scaler import MaxAbsScaler, Scaler, StandarScaler
from torch_timeseries.datasets import ETTm1
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.nn.Informer import Informer
from torch.nn import MSELoss, L1Loss, GRU
from omegaconf import OmegaConf
import torch.nn as nn
from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.nn.metric import R2, Corr


class GRU(nn.Module):
    def __init__(
        self, n_nodes, input_seq_len, casting_dim, dropout, out_len, num_layers=2
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_nodes,
            hidden_size=casting_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.time_decoder = nn.Linear(input_seq_len, out_len)
        self.node_layer = nn.Linear(casting_dim, n_nodes)

    def forward(self, input):
        output, _ = self.gru(input)  # (batch, seq_len, hidden_dim)
        output = output.transpose(1, 2)
        output = self.time_decoder(output)  # (batch, hidden_dim, out_len)
        output = output.transpose(1, 2)
        output = self.node_layer(output)  # (batch, out_len, n_nodes)
        output = output.squeeze(1)  # (B, N) or (B, out_len, N)
        return output


@dataclass
class GRUExperiment(Experiment):
    model_type: str = "GRU"
    hidden_size: int = 128
    dropout: float = 0.1
    num_layers: float = 2
    invtrans_loss: bool = False

    def _init_model_optm(self):
        self.model = GRU(
            self.dataset.num_features,
            self.windows,
            self.hidden_size,
            dropout=self.dropout,
            num_layers=self.num_layers,
            out_len=self.pred_len,
        )
        self.model = self.model.to(self.device)

        self.model_optim = self._parse_type(self.optm_type)(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        outputs = self.model(batch_x)  # (B, N) or (B, out_len, N)
        preds = self.scaler.inverse_transform(outputs)
        batch_y = self.scaler.inverse_transform(batch_y)
        if self.invtrans_loss:
            preds = self.scaler.inverse_transform(outputs)
            batch_y = self.scaler.inverse_transform(batch_y)
        else:
            preds = outputs
            batch_y = batch_y

        return preds.squeeze(), batch_y.squeeze()


def main():
    exp = GRUExperiment(
        dataset_type="ETTh1",
        data_path="./data",
        optm_type="Adam",
        batch_size=64,
        device="cuda:1",
        windows=96,
        hidden_size=64,
        horizon=3,
        epochs=100,
        lr=0.001,
        pred_len=1,
        seed=1,
        scaler_type="MaxAbsScaler",
        wandb=False,
    )

    exp.run()


if __name__ == "__main__":
    main()
