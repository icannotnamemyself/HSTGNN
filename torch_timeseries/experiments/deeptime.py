from typing import Dict, List, Type
import fire
import numpy as np
import torch
from tqdm import tqdm
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.models import DeepTIMe
import wandb
from typing import List

from dataclasses import dataclass, asdict, field


@dataclass
class DeepTIMeExperiment(Experiment):
    model_type: str = "DeepTIMe"
    
    
    layer_size : int = 256
    inr_layers : int = 5
    n_fourier_feats : int = 4096
    scales : List[int] = field(default_factory=lambda:[0.01, 0.1, 1, 5, 10, 20, 50, 100])
    freq : str = 'h'
    
    def _init_model(self):
        
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}

        self.model = DeepTIMe(
            datetime_feats=freq_map[self.freq],
            layer_size=self.layer_size,
            inr_layers=self.inr_layers,
            n_fourier_feats=self.n_fourier_feats,
            scales=self.scales
        )
        self.model = self.model.to(self.device)


    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.to(self.device).float()
        batch_x_mark = batch_x_mark.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_y_mark = batch_y_mark.to(self.device).float()
        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
        return outputs, batch_y


    def _init_data_loader(self):
        self.dataset : TimeSeriesDataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        self.dataloader = ChunkSequenceTimefeatureDataLoader(
            self.dataset,
            self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            scale_in_train=False,
            shuffle_train=True,
            freq=self.freq,
            batch_size=self.batch_size,
            train_ratio=0.7,
            val_ratio=0.2,
            num_worker=self.num_worker,
        )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = self.dataloader.train_size
        self.val_steps = self.dataloader.val_size
        self.test_steps = self.dataloader.test_size

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")


def main():
    exp = DeepTIMeExperiment(epochs=1,dataset_type="ExchangeRate", windows=96)
    exp.run()


if __name__ == "__main__":
    # fire.Fire(LaSTExperiment)
    main()
    
    
