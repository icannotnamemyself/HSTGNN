import fire
import torch
import torch.nn as nn

from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.nn.embedding import *

from torch_timeseries.datasets.dataset import TimeSeriesDataset
from tqdm import tqdm
from dataclasses import asdict, dataclass


@dataclass
class SoftMLPExpertExperiment(Experiment):
    model_type = 'SoftMLPExpert'
    experts_num : int = 5
    normalization : bool = True
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
        outputs = self.model(batch_x)  # torch.Size([batch_size, num_nodes])
        return outputs, batch_y


    def _init_model(self):
        self.model = AdaptiveFusion(
            # N, T, O
            self.dataset.num_features,
            self.windows,
            self.pred_len,
            experts_num = self.experts_num,
            normalization=self.normalization
        )
        self.model = self.model.to(self.device)


class WeightingModule(nn.Module):
    def __init__(self, N, T, O, expert_num) -> None:
        super(WeightingModule, self).__init__()
        self.N = N
        self.T = T
        self.expert_num = expert_num
        self.weighting_model =  nn.Sequential(
                    nn.Linear(T, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

        self.output =  nn.Sequential(
                nn.Linear(N, 16),
                nn.ReLU(),
                nn.Linear(16, expert_num),
            )
        

    def forward(self, x):
        # x : (B, T, N)
        B = x.size(0)
        
        weights = self.weighting_model(x.transpose(1,2)).squeeze(2) # (B, T, 1)
        output = self.output(weights) # B, E
        
        expert_weight = torch.softmax(output, dim=1)
        return expert_weight # (B, E)
        
        
        
class AdaptiveFusion(nn.Module):
    def __init__(self, N, T, O, experts_num = 4, normalization=True) -> None:
        super(AdaptiveFusion, self).__init__()
        self.normalization = normalization

        self.weight_module = WeightingModule(N, T, O, experts_num)
        self.pred_len = O
        self.expert_models = nn.ModuleList()
        self.experts_num = experts_num
        
        for i in range(experts_num):
            layers = nn.Sequential(
                nn.Linear(T, 128),
                nn.ReLU(),
            )
            for j in range(i):
                layers.append( 
                    nn.Sequential(
                        nn.Linear(128, 32),
                        nn.ReLU(),
                        nn.Linear(32, 128),
                        nn.ReLU(),
                    )
                )
            layers.append(
                    nn.Sequential(
                        nn.Linear(128, O),
                    )
            )
            self.expert_models.append(layers)
            
        
    def forward(self, x):
        # x : (B, T, N)
        batch_size = x.size(0)
        N = x.size(2)
        if self.normalization:
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last
        
        weights = self.weight_module(x)
        expert_results = []
        for i in range(self.experts_num):
            expert_results.append(self.expert_models[i](x.transpose(1,2)))
        expert_results = torch.stack(expert_results, dim=3) # (B N O E
        results= torch.einsum('bnoe, be -> bno', expert_results, weights)
        results = results.transpose(1,2)
        if self.normalization:
            results = results + seq_last
        return results # .transpose(1,2) # (B, O, N)


if __name__ == "__main__":
    fire.Fire(SoftMLPExpertExperiment)


