import torch
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset, TimeSeriesStaticGraphDataset
from torch_timeseries.experiments.experiment import Experiment
from torch_geometric_temporal.nn.attention import GMAN

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.utils.adj import adj_to_edge_index_weight

@dataclass
class GMANExperiment(Experiment):
    model_type: str = "GMAN1"
    L : int = 3
    K : int = 8
    d : int = 8 
    bn_decay : float = 0.1
    steps_per_day : int = 288
    use_bias : bool = True
    mask : bool = False
    
        
    def _init_model(self):
        assert isinstance(self.dataset, TimeSeriesStaticGraphDataset), "dataset must be of type TimeSeriesStaticGraphDataset"
        # adj = torch.tensor(self.dataset.adj).to(self.device, dtype=torch.float32)
        self.model = GMAN(
            L=self.L,
            K=self.K,
            d=self.d,
            num_his=self.windows,
            bn_decay=self.bn_decay,
            steps_per_day=self.pred_len,
            use_bias=self.use_bias,
            mask=self.mask,
        )
        self.model = self.model.to(self.device)
        
        self.spatial_embedding = torch.nn.Embedding(self.dataset.num_features, self.d*self.K).to(self.device)
        self.temporal_embedding = torch.nn.Embedding(self.windows + self.pred_len, 2).to(self.device)
        
        self.edge_index, self.edge_weight = adj_to_edge_index_weight(self.dataset.adj)
        self.edge_index = torch.tensor(self.edge_index).to(self.device)
        self.edge_weight = torch.tensor(self.edge_weight).to(self.device)


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
        outputs = self.model(batch_x,self.spatial_embedding.weight,self.temporal_embedding.weight.expand(batch_size, -1, -1))  # torch.Size([batch_size, out_seq_len, num_nodes])

        if self.pred_len == 1:
            return outputs.squeeze(1), batch_y.squeeze(1)
        elif self.pred_len > 1:
            return outputs, batch_y
        return outputs, batch_y


def main():
    exp = GMANExperiment(
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




def cli():
    import fire
    fire.Fire(GMANExperiment)

if __name__ == "__main__":
    cli()

