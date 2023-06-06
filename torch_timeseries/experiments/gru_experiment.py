import random
import time
from typing import Dict, Type
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.data.scaler import MaxAbsScaler, Scaler, StandarScaler
from torch_timeseries.datasets import ETTm1
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.nn.Informer import Informer
from torch.nn import MSELoss, L1Loss, GRU
from omegaconf import OmegaConf
import torch.nn as nn
from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.nn.metric import R2, Corr


class MyGRU(nn.Module):
    def __init__(self, n_nodes,input_seq_len, casting_dim, dropout, out_len, num_layers=2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_nodes, hidden_size=casting_dim,num_layers=num_layers, batch_first=True,   dropout=dropout
        )

        self.time_decoder = nn.Linear(input_seq_len, out_len)
        self.node_layer = nn.Linear(casting_dim, n_nodes)
    
    def forward(self, input):

        output, _ = self.gru(input)  # (batch, seq_len, hidden_dim)
        output = output.transpose(1, 2)
        output = self.time_decoder(output)  # (batch, hidden_dim, out_len)
        output = output.transpose(1, 2)
        output = self.node_layer(output)  # (batch, out_len, n_nodes)
        output = output.squeeze(1)  # (B, N) or (B, out_len, N)
        return output

@dataclass
class GRUExperiment(Experiment):
    model_type: str = "GRU"
    hidden_size :int = 512
    dropout:float = 0.1
    num_layers:float = 2
    
    
    def config_wandb(self):
        if self.wandb is True:
            run = wandb.init(
                project="GRU",
                name="MyfirstRun",
                notes="test first run",
                tags=["baseline", "informer"],
            )
            wandb.config.update(asdict(self))

    def _init_model_optm_loss(self):
        self.model = MyGRU(self.dataset.num_features,self.windows, self.hidden_size, dropout=self.dropout,num_layers=self.num_layers,out_len=self.pred_len)
        self.model = self.model.to(self.device)

        self.model_optim = self._parse_type(self.optm_type)(
            self.model.parameters(), lr=self.lr,weight_decay=self.l2_weight_decay
        )
        self.loss_func = MSELoss()

    def init(self):
        self.reproducible()

        self.config_wandb()
        #  2. Capture a dictionary of hyperparameters

        self._init_data_loader()

        self._init_model_optm_loss()

        self._init_checkpoint()

        self.metrics = MetricCollection(
            metrics={
                "r2": R2(self.dataset.num_features, multioutput="uniform_average"),
                "r2_wrighted": R2(
                    self.dataset.num_features, multioutput="variance_weighted"
                ),
                "mse": MeanSquaredError(),
                "corr": Corr(),
                # "trend_acc" : TrendAcc()
            }
        ).to(self.device)

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        return total_loss

    def test():
        pass

    def evaluate(self, dataloader):
        # 训练模型...
        self.model.eval()
        self.metrics.reset()
        for batch_x, batch_y, batch_x_date_enc, batch_y_date_enc in dataloader:
            preds, truths = self._process_one_batch(
                batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
            )
            self.metrics.update(preds, truths)

        val_res_str = " | ".join(
            [
                f"{name}: {round(float(metric.compute()), 4)}"
                for name, metric in self.metrics.items()
            ]
        )
        return val_res_str

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        outputs  = self.model(batch_x)  # (B, N) or (B, out_len, N)
        preds = self.dataset.inverse_transform(outputs)
        batch_y = self.dataset.inverse_transform(batch_y)

        return preds.squeeze(), batch_y.squeeze()

    def test():
        pass

    def train(self):
        pass

    def run(self):
        self.init()

        print(
            f"model parameters: {sum([p.nelement() for p in self.model.parameters()])}"
        )
        epoch_time = time.time()
        for epoch in range(self.epochs):
            train_loss = []

            with torch.enable_grad(), tqdm(total=self.train_steps) as progress_bar:
                self.model.train()
                for i, (
                    batch_x,
                    batch_y,
                    batch_x_date_enc,
                    batch_y_date_enc,
                ) in enumerate(self.train_loader):
                    self.model_optim.zero_grad()

                    pred, true = self._process_one_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                    )
                    loss = self.loss_func(pred, true)

                    loss.backward()
                    self.model_optim.step()
                    
                    train_loss.append(loss.item())

                    progress_bar.update(self.batch_size)
                    progress_bar.set_postfix(
                        epoch=epoch,
                        loss=loss.item(),
                        lr=self.model_optim.param_groups[0]["lr"],
                        refresh=True,
                    )

                    if self.wandb:
                        wandb.log({"loss": loss.item()}, step=i)


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            
            val_res_str = self.evaluate(self.train_loader)
            print(val_res_str)

            self._save(epoch=epoch)

            # if self.wandb:
            #     wandb.run.summary["best_accuracy"] = test_accuracy
            #     wandb.log({"vali_loss": vali_loss, "test_loss": test_loss}, step=epoch)

            # print(
            #     "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #         epoch + 1, self.train_steps, train_loss, vali_loss, test_loss
            #     )
            # )


def main():
    exp = GRUExperiment(
        dataset_type="ETTh1",
        data_path="./data",
        optm_type="Adam",
        batch_size=64,
        device="cuda:1",
        windows=96,
        hidden_size=64,
        horizon=3,
        epochs=100,
        lr=0.001,
        pred_len=1,
        seed=1,
        scaler_type="MaxAbsScaler",
        wandb=False,
    )

    exp.run()


if __name__ == "__main__":
    main()
