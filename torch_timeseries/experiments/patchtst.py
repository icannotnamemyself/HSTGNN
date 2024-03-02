import random
import time
from typing import Dict, Type
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.data.scaler import MaxAbsScaler, Scaler
from torch_timeseries.datasets import ETTm1
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.models import PatchTST
from torch.nn import MSELoss, L1Loss

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.nn.metric import R2, Corr
import fire


@dataclass
class PatchTSTExperiment(Experiment):
    model_type: str = "PatchTST"
    
    d_model: int = 512
    e_layers: int = 2
    d_ff: int = 512  # out of memoery with d_ff = 2048
    dropout: float = 0.0
    n_heads : int = 8
    patch_len : int = 16
    stride : int = 8
    label_len :int = 48
    
    
    def _init_model(self):
        self.model = PatchTST(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            dropout=self.dropout,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            task_name="long_term_forecast",
            num_class=0
            )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()

        
        # no decoder input
        # label_len = 1
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.label_len:, :], batch_y_date_enc], dim=1
        )
        outputs = self.model(batch_x, batch_x_date_enc,
                             dec_inp, dec_inp_date_enc)
        return outputs, batch_y



def main():
    exp = PatchTSTExperiment(dataset_type="ExchangeRate", windows=96)

    exp.run()

if __name__ == "__main__":
    fire.Fire(PatchTSTExperiment)
