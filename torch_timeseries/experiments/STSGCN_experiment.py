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
from torch_timeseries.models.STSGCN import STSGCN
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
from torch_geometric_temporal.nn.attention import GMAN

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict, field

from torch_timeseries.utils.adj import adj_to_edge_index_weight

def construct_adj(A, steps):
    """
    构建local 时空图
    :param A: np.ndarray, adjacency matrix, shape is (N, N)
    :param steps: 选择几个时间步来构建图
    :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    """
    N = len(A)  # 获得行数
    adj = np.zeros((N * steps, N * steps))

    for i in range(steps):
        """对角线代表各个时间步自己的空间图，也就是A"""
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            """每个节点只会连接相邻时间步的自己"""
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1

    return adj

@dataclass
class STSGCNExperiment(Experiment):
    model_type: str = "STSGCN"
    hidden_dims : List[int] = field(default_factory=lambda:[[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]])
    first_layer_embedding_size : int = 32
    out_layer_dim : int = 32
    use_mask : bool = True
    temporal_emb : bool = True
    spatial_emb : bool = True
    activation : str = 'relu'
    adj_steps : int = 3
    
        
    def _init_model(self):
        assert isinstance(self.dataset, TimeSeriesStaticGraphDataset), "dataset must be of type TimeSeriesStaticGraphDataset"
        # adj = torch.tensor(self.dataset.adj).to(self.device, dtype=torch.float32)
        local_adj = construct_adj(A=self.dataset.adj, steps=self.adj_steps)
        local_adj = torch.tensor(local_adj).to(self.device).float()
        self.model = STSGCN(
            adj=local_adj,
            history=self.windows,
            num_of_vertices=self.dataset.num_features,
            in_dim=1,
            hidden_dims=self.hidden_dims,
            first_layer_embedding_size=self.first_layer_embedding_size,
            out_layer_dim=self.out_layer_dim,
            activation=self.activation,
            use_mask=self.use_mask,
            temporal_emb=self.temporal_emb,
            spatial_emb=self.spatial_emb,
            horizon=self.pred_len,
            strides=self.adj_steps
        )        
        self.model = self.model.to(self.device)
        

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_size = batch_x.shape[0]
        batch_x = batch_x.to(self.device, dtype=torch.float32) # (B ,T , N)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device)
        batch_y_date_enc = batch_y_date_enc.to(self.device)
        outputs = self.model(batch_x.unsqueeze(-1))  # torch.Size([batch_size, out_seq_len, num_nodes])

        if self.pred_len == 1:
            return outputs.squeeze(1), batch_y.squeeze(1)
        elif self.pred_len > 1:
            return outputs, batch_y
        return outputs, batch_y


def main():
    exp = STSGCNExperiment(
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
