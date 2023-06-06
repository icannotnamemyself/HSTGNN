import os
import random
import time
from typing import Dict, Type, Union
import numpy as np
import torch
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceRandomSplitter, SequenceSplitter
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
    horizon: int = 3
    batch_size: int = 64
    pred_len: int = 1
    data_path: str = "./data"
    device: str = "cuda:0"
    windows: int = 368
    lr: float = 0.0003
    seed: int = 42
    epochs: int = 1
    wandb: bool = True

    model_type: str = "Informer"
    load_model_path: str = "./results"
    save_dir: str = "./results"
    l2_weight_decay: float = 0
    experiment_label: str = str(int(time.time()))


class Experiment(Settings):
    def init(self):
        assert self.pred_len >= self.label_len

        self.reproducible()
        self._init_data_loader()

        self._init_model_optm_loss()

        self._init_checkpoint()

    def _init_data_loader(self):
        self.dataset = MultiStepTimeFeatureSet(
            self._parse_type(self.dataset_type)(root=self.data_path),
            self._parse_type(self.scaler_type)(),
            horizon=self.horizon,
            window=self.windows,
            steps=self.pred_len,
            freq="h",
        )
        self.srs = SequenceRandomSplitter(
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            batch_size=self.batch_size,
            shuffle_train=True,
        )
        self.train_loader, self.val_loader, self.test_loader = self.srs(self.dataset)
        self.train_steps = self.srs.train_size
        self.val_steps = self.srs.val_size
        self.test_steps = self.srs.test_size

    def _init_model_optm_loss(self):
        self.model = self._parse_type(self.model_type)().to(self.device)
        self.model_optim = self._parse_type(self.optm_type)(
            self.model.parameters(), lr=self.lr
        )

        self.loss_func = MSELoss()

    def _init_checkpoint(self):
        # self.checkpoint_path  = os.path.join( os.path.join(os.path.join(self.save_dir, f"{self.model_type}"), self.dataset_type), self.experiment_label)
        self.checkpoint_path = os.path.join(
            self.save_dir, f"{self.model_type}/{self.dataset_type}"
        )
        self.checkpoint_filepath = os.path.join(
            self.checkpoint_path, f"{self.experiment_label}.pth"
        )
        self.app_state = {
            "model": self.model,
            "optimizer": self.model_optim,
        }

    def _parse_type(self, str_or_type: Union[Type, str]) -> Type:
        if isinstance(str_or_type, str):
            return eval(str_or_type)
        elif isinstance(str_or_type, type):
            return str_or_type
        else:
            raise RuntimeError(f"{str_or_type} should be string or type")

    def _save(self, epoch):
        self.app_state.update({"epoch": epoch})

        # 检查目录是否存在
        if not os.path.exists(self.checkpoint_path):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.checkpoint_path)
            print(f"Directory '{self.checkpoint_path}' created successfully.")

        torch.save(self.app_state, self.checkpoint_filepath)

    def _load_from_checkpoint(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = checkpoint["model"]
        self.model_optim = checkpoint["optimizer"]

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
        batch_y = self.dataset.inverse_transform(batch_y)

        return pred.squeeze(), batch_y.squeeze()

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
        device="cuda:3",
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
