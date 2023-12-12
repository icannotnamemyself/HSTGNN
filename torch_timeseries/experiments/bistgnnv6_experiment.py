import hashlib
import random
import time
from typing import Dict, List, Type
import numpy as np
import torch
import json
import os
from tqdm import tqdm
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset, TimeSeriesStaticGraphDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.models import  BiSTGNNv6
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset


from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict, field


@dataclass
class BiSTGNNv6Experiment(Experiment):
    model_type: str = "BiSTGNNv6"
    
    remain_prob:float=1.0
    gcn_type:str='weighted_han'
    graph_build_type:str='adaptive'
    output_layer_type:str='tcn8'
    
    latent_dim:int=32
    gcn_layers:int=2
    tn_layers:int=1
    
    heads:int = 1
    negative_slope:float = 0.2
    
    node_static_embed_dim :int = 16
    dropout:float=0.0
    act:str='elu'
    tcn_channel:int=16
    rebuild_time:bool=True
    rebuild_space:bool=True
    tcn_layers:int = 5
    dilated_factor:int = 2
    self_loop_eps:float= 0.1
    without_tn_module:bool = False
    without_gcn:bool = False
    d0 : int = 1
    kernel_set : List[int] = field(default_factory=lambda:[2,3,6,7])
    
    def reproducible(self, seed):
        # for reproducibility
        # torch.set_default_dtype(torch.float32)
        print("torch.get_default_dtype()", torch.get_default_dtype())
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determinstic = True


    def _run_identifier(self, seed) -> str:
        ident = self.result_related_config
        ident["seed"] = seed
        # only influence the evluation result, not included here
        ident['invtrans_loss'] = False
        
        if self.output_layer_type != 'tcn8':
            del ident['d0'] 
            del ident['kernel_set']
            print("delete d0")

        ident_md5 = hashlib.md5(
            json.dumps(ident, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return str(ident_md5)


    
    def _init_model(self):
        predefined_NN_adj = None
        padded_A = None
        if isinstance(self.dataset, TimeSeriesStaticGraphDataset) and self.pred_len > 1:
            predefined_NN_adj = torch.tensor(self.dataset.adj).to(self.device)
            D = torch.diag(torch.sum(predefined_NN_adj, dim=1))
            D_sqrt_inv = torch.sqrt(torch.inverse(D))
            normalized_predefined_adj = D_sqrt_inv @predefined_NN_adj @ D_sqrt_inv
            padded_A = torch.nn.functional.pad(normalized_predefined_adj, (0, self.windows, 0, self.windows), mode='constant', value=0).float()

        else:
            padded_A = None

        if isinstance(self.dataset, PeMS_D7):
            temporal_embed_dim = 0
        else:
            temporal_embed_dim = 4
        
        self.model = BiSTGNNv6(
            seq_len=self.windows,
            num_nodes=self.dataset.num_features,
            temporal_embed_dim=temporal_embed_dim, # 4 for hour embedding
            latent_dim=self.latent_dim,
            predefined_adj=padded_A,
            heads=self.heads,
            negative_slope=self.negative_slope,
            gcn_layers=self.gcn_layers,
            dropout=self.dropout,
            graph_build_type=self.graph_build_type,
            graph_conv_type=self.gcn_type,
            tn_layers=self.tn_layers,
            rebuild_time=self.rebuild_time,
            rebuild_space=self.rebuild_space,
            out_seq_len=self.pred_len,
            node_static_embed_dim=self.node_static_embed_dim,
            tcn_layers=self.tcn_layers,
            dilated_factor=self.dilated_factor,
            tcn_channel=self.tcn_channel,
            act=self.act,
            output_layer_type=self.output_layer_type,
            self_loop_eps=self.self_loop_eps,
            without_tn_module=self.without_tn_module,
            kernel_set=self.kernel_set
        )
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
        batch_x_date_enc = batch_x_date_enc.to(self.device, dtype=torch.float32)
        batch_y_date_enc = batch_y_date_enc.to(self.device, dtype=torch.float32)
        batch_x = batch_x.transpose(1,2)
        outputs = self.model(batch_x,batch_x_date_enc)  # torch.Size([batch_size, num_nodes])
        # single step prediction
        return outputs, batch_y


def main():
    exp = BiSTGNNv6Experiment(
        dataset_type="ExchangeRate",
        data_path="./data",
        optm_type="Adam",
        batch_size=64,
        device="cuda:0",
        windows=168,
        output_layer_type='tcn6',
        heads=2,
    )
    # exp.config_wandb(
    #     "project",
    #     "name"
    # )
    exp.run()



if __name__ == "__main__":
    main()
