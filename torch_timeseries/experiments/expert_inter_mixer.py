import fire
import torch
import torch.nn as nn

from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.nn.embedding import *

from torch_timeseries.datasets.dataset import TimeSeriesDataset
from tqdm import tqdm
from dataclasses import asdict, dataclass

class LinearExpert(nn.Module):
    def __init__(self, N, T, O, h=32, mode='s') -> None:
        super(LinearExpert, self).__init__()
        self.N = N
        self.T = T
        self.h = h
        self.mode = mode
        if mode == 't':
            self.ffn = nn.Sequential(
                    nn.Linear(T, h),
                    nn.ReLU(),
                    nn.Linear(h, T),
            )
        elif mode == 's':
            self.ffn = nn.Sequential(
                    nn.Linear(N, h),
                    nn.ReLU(),
                    nn.Linear(h, N),
            )

    def forward(self, x):
        # x: (B, T, N) 
        if self.mode == 't':
            x = x.transpose(1,2) #  (B,  N, T) 
            x1 = self.ffn(x)  #  (B,  N, T) 
            return x1.transpose(1,2)
            
        elif self.mode == 's':
            x1 = self.ffn(x)  # (B, T, N) 
            return x1

        
class ExpertLayer(nn.Module):
    def __init__(self, N, T, O, layer_num = 1, ht=32, hs=32, elementwise_affine=False) -> None:
        super(ExpertLayer, self).__init__()
        
        self.N = N
        self.T = T
        self.ht = ht
        self.hs = hs
        self.elementwise_affine = elementwise_affine
        
        self.layer_num = layer_num
        
        self.models = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.residuel_models = nn.ModuleList()
        
        
        self.skip_models = nn.ModuleList()
        self.skip_end = nn.Sequential(
                nn.Linear(T, 32),
                nn.ReLU(),
                nn.Linear(32, T),
            )
        
        for i in range(self.layer_num):
            if i % 2 ==0:
                self.models.append(LinearExpert(N, T, O, ht, 't'))
            else:
                self.models.append(LinearExpert(N, T, O, hs, 's'))
            self.norms.append(nn.LayerNorm([T], elementwise_affine=elementwise_affine))
            self.skip_models.append(LinearExpert(N, T, O, ht, 't'))
            self.residuel_models.append(LinearExpert(N, T, O, ht, 't'))

        self.output =  nn.Sequential(
                nn.Linear(T, 32),
                nn.ReLU(),
                nn.Linear(32, O),
            )
        
        
    def forward(self, x):
        # B T N
        
        skip = x
        for i in range(self.layer_num):
            x  = self.models[i](x) # B T N
            skip = self.skip_models[i](skip) + x # B T N
            x = x + self.residuel_models[i](x) # B T N
            # if i % 2 ==0:
            #     x = self.norms[i](x.transpose(1,2)).transpose(1,2) # B T N
            # else:
            #     x = self.norms[i](x) # B T N
                
            x = self.norms[i](x.transpose(1,2)).transpose(1,2) # B T N
                
                
        out = self.output(x.transpose(1,2) + self.skip_end(skip.transpose(1,2)))  # B N O
        return out.transpose(1,2) # B O N

class WeightingModule(nn.Module):
    def __init__(self, N, T, O, expert_num, mode='softmax') -> None:
        super(WeightingModule, self).__init__()
        self.N = N
        self.T = T
        self.expert_num = expert_num
        self.mode = mode
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
        
        
        if self.mode == 'softmax':
            expert_weight = torch.softmax(output, dim=1)
        elif self.mode == 'output':
            expert_weight =  output
        elif self.mode == 'relu':
            expert_weight = torch.relu(output)
        return expert_weight # (B, E)
        
        
        
class ExpertLayerWithResidual(nn.Module):
    def __init__(self, N, T, O, experts_num = 4, base_layer=1, weighting = True, weight_mode='output', ht=32, hs=32, last_model=False, normalization=True, elementwise_affine = False) -> None:
        super(ExpertLayerWithResidual, self).__init__()
        self.normalization = normalization
        self.weighting = weighting
        if weighting:
            self.weight_module = WeightingModule(N, T, O, experts_num, weight_mode)
        self.pred_len = O
        self.last_model = last_model
        self.base_layer = base_layer
        self.expert_models = nn.ModuleList()
        self.experts_num = experts_num
        
        for i in range(experts_num):
                self.expert_models.append(ExpertLayer(N, T, O, i+1 + self.base_layer, ht, hs, elementwise_affine))
                
            
        
    def forward(self, x):
        # x : (B, T, N)
        # 用于收集每个序列处理后的结果
        batch_size = x.size(0)
        N = x.size(2)
        if self.normalization:
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last
            
        results = None
        expert_results = []
        if self.weighting:
            weights = self.weight_module(x)
            
            for i in range(self.experts_num):
                expert_results.append(self.expert_models[i](x))
            expert_results = torch.stack(expert_results, dim=3) if not self.last_model else expert_results[-1].unsqueeze(3)  # (B N O E
            # import pdb;pdb.set_trace()
            results= torch.einsum('bnoe, be -> bno', expert_results, weights)
        else:
            for i in range(self.experts_num):
                expert_results.append(self.expert_models[i](x))
            expert_results = torch.stack(expert_results, dim=3) if not self.last_model else expert_results[-1].unsqueeze(3)  # (B N O E
            results = expert_results.sum(3)
            
        if self.normalization:
            results = results + seq_last
        return results # .transpose(1,2) # (B, O, N)

@dataclass
class ExpertLayerWithResidualExperiment(Experiment):
    model_type : str = 'ExpertLayerWithResidual'
    elementwise_affine : bool = True
    experts_num : int = 5
    weighting : bool = False
    last_model : bool = False
    ht : int = 32
    weight_mode : str = 'output'
    base_layer : int = 0
    hs : int = 32
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
        self.model = ExpertLayerWithResidual(
            # N, T, O
            self.dataset.num_features,
            self.windows,
            self.pred_len,
            ht=self.ht,
            base_layer=self.base_layer,
            hs=self.hs,
            last_model=self.last_model,
            weighting=self.weighting,
            experts_num = self.experts_num,
            normalization=self.normalization,
            elementwise_affine=self.elementwise_affine,
            weight_mode=self.weight_mode,
        )
        self.model = self.model.to(self.device)



if __name__ == "__main__":
    fire.Fire(ExpertLayerWithResidualExperiment)


