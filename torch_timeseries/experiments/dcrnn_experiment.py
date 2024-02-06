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
from torch_timeseries.models import BiSTGNN, BiSTGNNv2
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
from torch_timeseries.models.DCRNN import DCRNN 

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.utils.adj import adj_to_edge_index_weight


@dataclass
class DCRNNExperiment(Experiment):
    model_type: str = "DCRNN1"
    K : int = 2
    enc_input_dim : int = 2
    dec_intput_dim : int = 1
    num_rnn_layers : int = 2
    rnn_units : int = 32
    
        
    def _init_model(self):
        assert isinstance(self.dataset, TimeSeriesStaticGraphDataset), "dataset must be of type TimeSeriesStaticGraphDataset"
        if isinstance(self.dataset, TimeSeriesStaticGraphDataset) and self.pred_len > 1:
            predefined_NN_adj = torch.tensor(self.dataset.adj).to(self.device)
            D = torch.diag(torch.sum(predefined_NN_adj, dim=1))
            D_sqrt_inv = torch.sqrt(torch.inverse(D))
            normalized_predefined_adj = D_sqrt_inv @predefined_NN_adj @ D_sqrt_inv


            self.model = DCRNN(adj_mat=normalized_predefined_adj.detach().cpu().numpy(), 
                               device=self.device,
                               enc_input_dim=self.enc_input_dim, dec_input_dim=self.dec_intput_dim, max_diffusion_step=self.K, num_nodes=self.dataset.num_features, num_rnn_layers=self.num_rnn_layers, rnn_units=self.rnn_units, seq_len =self.windows, output_dim=1, filter_type='laplacian')
            self.model = self.model.to(self.device)



    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_size = batch_x.size(0)
        
    
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        # batch_x = batch_x.transpose(1,2)  # (B, N, T)
        
        
        source = torch.stack([batch_x,batch_x],dim=-1 ) # (B, T N)
        target = torch.stack([batch_x,batch_x],dim=-1 ) # (B,, T N,1 )
        
        pred = self.model(source, target, batch_size) # (T, B, N)
         
        pred = pred.transpose(0 , 1).to(self.device) # (B, T, N)
        
        pred = pred[:, :self.pred_len, :] # (B, O, N)
         
        return pred, batch_y


def main():
    exp = DCRNNExperiment(
        dataset_type="DummyDatasetGraph",
        data_path="./data",
        optm_type="Adam",
        batch_size=31,
        device="cuda:2",
        pred_len=3,
        horizon=1,
        windows=12,
    )
    # exp.config_wandb(
    #     "project",
    #     "name"
    # )
    exp.run()



if __name__ == "__main__":
    main()
