import random
import time
from typing import Dict, Type
import numpy as np
import torch
from torch_timeseries.data.scaler import MaxAbsScaler, Scaler
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.nn.mtgnn import MTGNN
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict


@dataclass
class MTGNNExperiment(Experiment):
    model_type: str = "MTGNN"
    gcn_true : bool = True
    gcn_depth : int = 3
    dropout:float=0.3
    subgraph_size:int=20
    node_dim:int=40
    dilation_exponential:float=1
    conv_channels:int=32
    residual_channels:int=32
    skip_channels:int=64
    end_channels:int=128
    seq_length:int=12
    in_dim:int=1
    # out_dim:int=12
    layers:int=3
    propalpha: float=0.05
    tanhalpha : float=3
    layer_norm_affline : bool =True
    skip_layer : bool =True
    residual_layer: bool =True


    def config_wandb(self):
        if self.wandb is True:
            run = wandb.init(
                project="project",
                name="MTGNN",
                notes="MTGNN",
                tags=["baseline", "forcast"],
            )
            wandb.config.update(asdict(self))

    def init(self):

        self.reproducible()

        self.config_wandb()
        #  2. Capture a dictionary of hyperparameters

        self.dataset = MultiStepTimeFeatureSet(
            self._parse_type(self.dataset_type)(root=self.data_path),
            self._parse_type(self.scaler_type)(),
            horizon=self.hoziron,
            window=self.windows,
            steps=self.pred_len,
        )

        assert self.subgraph_size <= self.dataset.num_features, f"graph size {self.subgraph_size} have to be bigger than data columns :{self.dataset.num_features}"
        self.model = MTGNN(
            gcn_true=self.gcn_true,
            buildA_true=True,
            gcn_depth=self.gcn_depth,
            num_nodes=self.dataset.num_features,
            device=self.device,
            predefined_A=None,
            static_feat=None,
            dropout=self.dropout,
            subgraph_size=self.subgraph_size,
            node_dim=self.node_dim,
            dilation_exponential=self.dilation_exponential,
            conv_channels=self.conv_channels,
            residual_channels=self.residual_channels,
            skip_channels=self.skip_channels,
            end_channels=self.end_channels,
            seq_length=self.windows,
            in_dim=self.in_dim,
            out_dim=self.pred_len,
            layers=self.layers,
            propalpha=self.propalpha,
            tanhalpha=self.tanhalpha,
            layer_norm_affline=self.layer_norm_affline,
            skip_layer=self.skip_layer,
            residual_layer=self.residual_layer,
        )
        self.model = self.model.to(self.device)

        self.model_optim = self._parse_type(self.optm_type)(
            self.model.parameters(), lr=self.lr
        )

        self.srs = SequenceSplitter(batch_size=self.batch_size)
        self.train_loader, self.val_loader, self.test_loader = self.srs(self.dataset)
        self.train_steps = len(self.train_loader)
        self.loss_func = MSELoss()

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            loss = criterion(pred, true)
            total_loss.append(loss)
        total_loss = np.average(torch.Tensor(total_loss).cpu())
        return total_loss

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_x_date_enc = batch_x_date_enc.to(self.device)
        batch_y_date_enc = batch_y_date_enc.to(self.device)


        input_x = batch_x.unsqueeze(1) 
        input_x = input_x.transpose(2,3 )   # torch.Size([batch_size, 1, num_nodes, windows])
        outputs = self.model(input_x)
        
        
        pred = self.dataset.inverse_transform(outputs)
        pred = pred.squeeze()
        return pred, batch_y

    def test():
        pass

    def train(self):
        pass

    def run(self):
        self.init()
        train_loss = []
        criterion = MSELoss()
        epoch_time = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            for i, (batch_x, batch_y, batch_x_date_enc, batch_y_date_enc) in enumerate(
                self.train_loader
            ):
                self.model_optim.zero_grad()

                pred, true = self._process_one_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                loss = self.loss_func(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )

                if self.wandb:
                    wandb.log({"loss": loss.item()}, step=i)

                loss.backward()
                self.model_optim.step()
                break

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.val_loader, criterion)
            test_loss = self.vali(self.test_loader, criterion)

            # if self.wandb:
            #     wandb.run.summary["best_accuracy"] = test_accuracy
            #     wandb.log({"vali_loss": vali_loss, "test_loss": test_loss}, step=epoch)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, self.train_steps, train_loss, vali_loss, test_loss
                )
            )


def main():
    exp = MTGNNExperiment(
        dataset_type="ETTm1",
        data_path="./data",
        optm_type="Adam",
        batch_size=32,
        device="cuda:1",
        windows=10,
        epochs=1,
        lr=0.001,
        hoziron=3,
        pred_len=3,
        subgraph_size=6,
        seed=1,
        scaler_type="MaxAbsScaler",
        wandb=False
    )

    exp.run()


if __name__ == "__main__":
    main()
