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
from torch_timeseries.models import TSMixer
from torch.nn import MSELoss, L1Loss
from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class TSMixerExperiment(Experiment):
    model_type: str = "TSMixer"
    
    n_mixer : int = 8
    dropout :int  = 0.05
    
    
    
    def _init_model(self):
        self.model = TSMixer(
            L=self.windows,
            C = self.dataset.num_features,
            T=self.pred_len,
            n_mixer=self.n_mixer,
            dropout=self.dropout
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        outputs = self.model(batch_x)
        return outputs.squeeze(), batch_y.squeeze()




def main():
    exp = TSMixerExperiment(dataset_type="ExchangeRate", windows=96)

    exp.run()

def cli():
    import fire
    fire.Fire(TSMixerExperiment)

if __name__ == "__main__":
    cli()
