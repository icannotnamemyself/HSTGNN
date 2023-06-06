import random
import time
from typing import Dict, Type
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.data.scaler import MaxAbsScaler, Scaler
from torch_timeseries.datasets import ETTm1
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.nn.Informer import Informer
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class InformerExperiment(Experiment):
    model_type: str = "Informer"
    label_len: int = 2

    factor: int = 5
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layer: int = 512
    d_ff: int = 512
    dropout: float = 0.0
    attn: str = "prob"
    embed: str = "fixed"
    activation = "gelu"
    distil: bool = True
    mix: bool = True

    def config_wandb(self):
        if self.wandb is True:
            run = wandb.init(
                project="informer",
                name="MyfirstRun",
                notes="test first run",
                tags=["baseline", "informer"],
            )
            wandb.config.update(asdict(self))

    def _init_model_optm_loss(self):
        self.model = Informer(
            self.dataset.num_features,
            self.dataset.num_features,
            self.dataset.num_features,
            self.pred_len,
            factor=self.factor,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            attn=self.attn,
            embed=self.embed,
            activation=self.activation,
            distil=self.distil,
            mix=self.mix,
        )
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

        # metric_computes = list(map(lambda key: (key,self.metrics[key].compute()), self.metrics.keys()))
        # import pdb; pdb.set_trace()
        val_res_str = " | ".join(
            [
                f"{name}: {round(float(metric.compute()), 4)}"
                for name, metric in self.metrics.items()
            ]
        )
        # val_res_str = " | ".join(["{}: {:.4f}±{:.4f}" for _ in self.metrics.keys()])

        return val_res_str

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_x_date_enc = batch_x_date_enc.to(self.device)
        batch_y_date_enc = batch_y_date_enc.to(self.device)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.label_len :, :], batch_y_date_enc], dim=1
        )
        outputs = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)
        pred = self.dataset.inverse_transform(outputs)
        batch_y = self.dataset.inverse_transform(batch_y)

        return pred.squeeze(), batch_y.squeeze()

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

                    if (i + 1) % 10 == 0:
                        print(
                            "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                                i + 1, epoch + 1, loss.item()
                            )
                        )
                    # Log info
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
            # vali_loss = self.vali(self.val_loader, criterion)
            # test_loss = self.vali(self.test_loader, criterion)

            # val_res_str = self.evaluate(self.val_loader)
            # print(val_res_str)
            
            
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
    exp = InformerExperiment(
        dataset_type="ETTh1",
        data_path="./data",
        optm_type="Adam",
        batch_size=64,
        device="cuda:1",
        windows=96,
        label_len=48,
        horizon=3,
        epochs=100,
        lr=0.001,
        dropout=0.05,
        d_ff=2048,
        pred_len=24,
        seed=1,
        scaler_type="MaxAbsScaler",
        wandb=False,
    )

    exp.run()


if __name__ == "__main__":
    main()
