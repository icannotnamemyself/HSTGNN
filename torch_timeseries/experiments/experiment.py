import random
import time
from typing import Dict, Type, Union
import numpy as np
import torch
from torch_timeseries.data.scaler import MaxAbsScaler, Scaler
from torch_timeseries.datasets import ETTm1
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.nn.Informer import Informer
from torch.nn import MSELoss
from omegaconf import OmegaConf

from torch.optim import Optimizer, Adam


from dataclasses import dataclass


@dataclass
class Settings:
    dataset_type: str
    optm_type: str = "Adam"
    scaler_type: str = "MaxAbsScaler"

    batch_size: int = 64
    pred_len: int = 3
    data_path: str = "./data"
    device: str = "cuda:0"
    windows: int = 168
    lr: float = 0.001
    seed: int = 1
    epochs: int = 1
    wandb: bool = False


class Experiment(Settings):
    def init(self):
        assert self.pred_len >= self.label_len

        self.reproducible()

        self.model = self._parse_type(self.model_type)().to(self.device)

        self.dataset = MultiStepTimeFeatureSet(
            self._parse_type(self.dataset_type)(root=self.data_path),
            self.scaler_type(),
            window=self.windows,
            steps=self.pred_len,
        )
        self.model_optim = self._parse_type(self.optm_type)(
            self.model.parameters(), lr=self.lr
        )

        self.srs = SequenceSplitter(batch_size=self.batch_size)
        self.train_loader, self.val_loader, self.test_loader = self.srs(self.dataset)
        self.train_steps = len(self.train_loader)
        self.loss_func = MSELoss()

    def _parse_type(self, str_or_type: Union[Type, str]) -> Type:
        if isinstance(str_or_type, str):
            return eval(str_or_type)
        elif isinstance(str_or_type, type):
            return str_or_type
        else:
            raise RuntimeError(f"{str_or_type} should be string or type")

    def reproducible(self):
        # for reproducibility
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

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

                loss.backward()
                self.model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.val_loader, criterion)
            test_loss = self.vali(self.test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, self.train_steps, train_loss, vali_loss, test_loss
                )
            )


def main():
    exp = Experiment(
        dataset_type="ETTm1",
        data_path="./data",
        optm_type="Adam",
        model_type="Informer",
        batch_size=32,
        device="cuda:1",
        windows=10,
        label_len=2,
        epochs=1,
        lr=0.001,
        pred_len=3,
        seed=1,
        scaler_type="MaxAbsScaler",
    )

    conf = OmegaConf.structured(exp)
    print(OmegaConf.to_yaml(conf))

    # exp = Experiment(settings)
    # exp.run()


if __name__ == "__main__":
    main()