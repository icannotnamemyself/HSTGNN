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
from torch_timeseries.models.STGCN import STGCN
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.utils.adj import adj_to_edge_index_weight
def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


@dataclass
class STGCNExperiment(Experiment):
    model_type: str = "STGCN1"
    stblock_num : int = 2
    Kt : int = 3
    Ks : int = 3
    act_func : str = 'glu'
    graph_conv_type : str = 'cheb_graph_conv'
        
    def _init_model(self):
        assert isinstance(self.dataset, TimeSeriesStaticGraphDataset), "dataset must be of type TimeSeriesStaticGraphDataset"
        
        blocks = []
        blocks.append([1])
        self.normalized_adj = torch.tensor(get_normalized_adj(self.dataset.adj)).to(self.device).float()
        
        Ko = self.windows - (self.Kt - 1) * 2 * self.stblock_num
        for l in range(self.stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([self.pred_len])

        self.model = STGCN(
            blocks,
            self.dataset.num_features,
            self.windows,
            self.normalized_adj,
            self.Ks,
            self.Kt,
            self.act_func,
            self.graph_conv_type,
            True, 
            0.5
        )
        self.model = self.model.to(self.device)
        
    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        # input (batch_size, num_nodes, num_timesteps, D)
        
        
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        # batch_x = batch_x.unsqueeze(-1)  # (B, T, N, D)
        batch_x = batch_x.unsqueeze(1)  # (B,D, T, N)
        y = self.model(batch_x).view(batch_size, self.pred_len, self.dataset.num_features) # (B, O, N)
        # y = self.model(batch_x).view(batch_size, self.) # (B N O)
        # y = y.transpose(1,2)  # (B, O, N)
        return y, batch_y


def main():
    exp = STGCNExperiment(
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
