import random
import time
from typing import Dict, Type
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.data.scaler import MaxAbsScaler, Scaler
from torch_timeseries.datasets import ETTm1
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.nn.Informer import Informer
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class InformerExperiment(Experiment):
    model_type: str = "Informer"
    label_len: int = 48

    factor: int = 5
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layer: int = 512
    d_ff: int = 512
    # TODO: fix dropout to 0.3
    dropout: float = 0.0
    attn: str = "prob"
    embed: str = "fixed"
    activation = "gelu"
    distil: bool = True
    mix: bool = True

    def _init_model(self):
        self.model = Informer(
            self.dataset.num_features,
            self.dataset.num_features,
            self.dataset.num_features,
            self.pred_len,
            factor=self.factor,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            attn=self.attn,
            embed=self.embed,
            activation=self.activation,
            distil=self.distil,
            mix=self.mix,
        )
        self.model = self.model.to(self.device)


    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_x_date_enc = batch_x_date_enc.to(self.device)
        batch_y_date_enc = batch_y_date_enc.to(self.device)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.label_len :, :], batch_y_date_enc], dim=1
        )
        outputs = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)
        return outputs, batch_y


def main():
    exp = InformerExperiment(
        dataset_type="ETTh1",
        data_path="./data",
        optm_type="Adam",
        batch_size=64,
        device="cuda:1",
        windows=96,
        label_len=48,
        horizon=3,
        epochs=100,
        lr=0.001,
        dropout=0.05,
        d_ff=2048,
        pred_len=24,
        seed=1,
        scaler_type="MaxAbsScaler",
        wandb=False,
    )

    exp.run()


if __name__ == "__main__":
    main()
