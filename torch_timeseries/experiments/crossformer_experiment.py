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
from torch_timeseries.nn.crossformer import Crossformer
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset


from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict


@dataclass
class CrossformerExperiment(Experiment):
    model_type: str = "Crossformer"
    
    seg_len:int = 4 # following crossformer paper,segment length from 4 to 24, samller segment yield better results
    win_size:int = 4  # default:2 , since winsize 2 is used in crossformer original paper
    factor:int = 10
    d_model:int=512
    d_ff:int = 512
    n_heads:int=8
    e_layers:int=3 
    dropout:float=0.0
    baseline = False

    def _init_model(self):
        self.model = Crossformer(
            data_dim=self.dataset.num_features, in_len=self.windows, out_len=self.pred_len, seg_len=self.seg_len, win_size = self.win_size,
                factor=self.factor, d_model=self.d_model, d_ff = self.d_ff, n_heads=self.n_heads, e_layers=self.e_layers, 
                dropout=self.dropout, baseline =self.baseline, device=self.device
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)
        return outputs, batch_y


def main():
    # import os
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    exp = CrossformerExperiment(
        dataset_type="ExchangeRate",
        data_path="./data",
        optm_type="Adam",
        batch_size=64,
        device="cuda:2",
        windows=96,
        d_ff=512,
        epochs=100,
        horizon=3,
        pred_len=1,
        scaler_type="StandarScaler",
    )
    # exp.config_wandb(
    #     "project",
    #     "name"
    # )
    exp.run(42)


if __name__ == "__main__":
    main()
