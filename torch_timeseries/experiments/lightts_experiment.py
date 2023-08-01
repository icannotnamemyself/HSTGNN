import random
import time
from typing import Dict, List, Type
import fire
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
from torch_timeseries.models import LightTS
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class LightTSExperiment(Experiment):
    model_type: str = "LightTS"
    
    d_model: int = 512
    dropout : float = 0.0
    chunk_size : int = 24
    
    
    
    
    def _init_model(self):
        self.model = LightTS(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            dropout=self.dropout,
            chunk_size=self.chunk_size,
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
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.pred_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.pred_len:, :], batch_y_date_enc], dim=1
        )
        outputs = self.model(batch_x, batch_x_date_enc,
                             dec_inp, dec_inp_date_enc)
        return outputs, batch_y



def main():
    exp = LightTSExperiment(dataset_type="ExchangeRate", windows=96)
    exp.run()


if __name__ == "__main__":
    fire.Fire(LightTSExperiment)
