import os
import random
import time
from typing import Dict, List, Type, Union
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection
from tqdm import tqdm
import wandb
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceRandomSplitter, SequenceSplitter
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.nn.Informer import Informer
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf

from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset


from dataclasses import asdict, dataclass

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class Settings:
    dataset_type: str
    optm_type: str = "Adam"
    scaler_type: str = "StandarScaler"
    horizon: int = 3
    batch_size: int = 64
    pred_len: int = 1
    data_path: str = "./data"
    device: str = "cuda:0"
    windows: int = 384
    lr: float = 0.0003
    # seed for experiment level randomness such as dataloader randomness
    seed: int = 42
    num_worker: int = 2
    epochs: int = 1
    wandb: bool = True
    loss_func_type: str = "mse"

    model_type: str = "Informer"
    load_model_path: str = "./results"
    save_dir: str = "./results"
    l2_weight_decay: float = 0.0005
    experiment_label: str = str(int(time.time()))


class Experiment(Settings):
    def config_wandb(
        self,
        project: str,
        name: str,
    ):
        if self.wandb is True:
            run = wandb.init(
                project=project,
                name=name,
            )
            wandb.config.update(asdict(self))
            print(f"running in config: {asdict(self)}")
        return self

    def config_wandb_verbose(
        self,
        project: str,
        name: str,
        tags: List[str],
        notes: str,
    ):
        if self.wandb is True:
            run = wandb.init(
                project=project,
                name=name,
                notes=notes,
                tags=tags,
            )
            wandb.config.update(asdict(self))
            print(f"running in config: {asdict(self)}")
        return self

    def _init_loss_func(self):
        loss_func_map = {"mse": MSELoss, "l1": L1Loss}
        self.loss_func = loss_func_map[self.loss_func_type]()

    def _init_metrics(self):
        self.metrics = MetricCollection(
            metrics={
                "r2": R2(self.dataset.num_features, multioutput="uniform_average"),
                "r2_weighted": R2(
                    self.dataset.num_features, multioutput="variance_weighted"
                ),
                "mse": MeanSquaredError(),
                "corr": Corr(),
                # "trend_acc" : TrendAcc()
            }
        ).to(self.device)

    def _init_data_loader(self):
        # self.dataset = MultiStepTimeFeatureSet(
        #     self._parse_type(self.dataset_type)(root=self.data_path),
        #     self._parse_type(self.scaler_type)(),
        #     horizon=self.horizon,
        #     window=self.windows,
        #     steps=self.pred_len,
        #     freq="h",
        # )
        # self.srs = SequenceRandomSplitter(
        #     train_ratio=0.7,
        #     val_ratio=0.2,
        #     test_ratio=0.1,
        #     batch_size=self.batch_size,
        #     shuffle_train=True,
        #     seed=self.seed,
        #     num_worker=self.num_worker
        # )
        self.dataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        self.dataloader = ChunkSequenceTimefeatureDataLoader(
            self.dataset,
            self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            scale_in_train=False,
            shuffle_train=True,
            freq="h",
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

    def _init_model_optm(self):
        self.model = self._parse_type(self.model_type)().to(self.device)
        self.model_optim = self._parse_type(self.optm_type)(
            self.model.parameters(), lr=self.lr
        )

    def _setup(self):
        # init data loader
        self._init_data_loader()

        # init metrics
        self._init_metrics()

        self.current_epochs = 0
        self.current_run = 0

        self.setuped = True

    def _setup_run(self, seed):
        # setup experiment  only once
        if not hasattr(self, "setuped"):
            self._setup()
        # setup torch and numpy random seed
        self.reproducible(seed)
        # init model, optimizer and loss function
        self._init_model_optm()
        # init loss function based on given loss func type
        self._init_loss_func()

    def _parse_type(self, str_or_type: Union[Type, str]) -> Type:
        if isinstance(str_or_type, str):
            return eval(str_or_type)
        elif isinstance(str_or_type, type):
            return str_or_type
        else:
            raise RuntimeError(f"{str_or_type} should be string or type")

    def _save(self, seed):
        self.checkpoint_path = os.path.join(
            self.save_dir, f"{self.model_type}/{self.dataset_type}"
        )
        self.checkpoint_filepath = os.path.join(
            self.checkpoint_path, f"{self.experiment_label}.pth"
        )
        # 检查目录是否存在
        if not os.path.exists(self.checkpoint_path):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.checkpoint_path)
            print(f"Directory '{self.checkpoint_path}' created successfully.")

        self.app_state = {
            "model": self.model,
            "optimizer": self.model_optim,
        }

        self.app_state.update(asdict(self))
        torch.save(
            self.app_state, f"{self.checkpoint_filepath}-{seed}-{self.current_epoch}"
        )

    def _load_from_checkpoint(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = checkpoint["model"]
        self.model_optim = checkpoint["optimizer"]

    def reproducible(self, seed):
        # for reproducibility
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determinstic = True

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
        pred = self.dataloader.scaler.inverse_transform(outputs)
        batch_y = self.dataloader.scaler.inverse_transform(batch_y)

        return pred.squeeze(), batch_y.squeeze()

    def _test(self) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        print("Evaluating .... ")
        with tqdm(total=self.test_steps) as progress_bar:
            for (
                batch_x,
                batch_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in self.test_loader:
                preds, truths = self._process_one_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                self.metrics.update(preds, truths)

                progress_bar.update(batch_x.shape[0])
        test_result = {
            name: float(metric.compute()) for name, metric in self.metrics.items()
        }
        if self.wandb is True:
            for name, metric_value in test_result.items():
                wandb.run.summary["test_" + name] = metric_value

        print(f"test_results: {test_result}")
        return test_result

    def _evaluate(self):
        self.model.eval()
        self.metrics.reset()
        print("Evaluating .... ")
        with tqdm(total=self.val_steps) as progress_bar:
            for batch_x, batch_y, batch_x_date_enc, batch_y_date_enc in self.val_loader:
                preds, truths = self._process_one_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                self.metrics.update(preds, truths)

                progress_bar.update(batch_x.shape[0])

        val_result = {
            name: float(metric.compute()) for name, metric in self.metrics.items()
        }

        if self.wandb is True:
            for name, metric_value in val_result.items():
                wandb.run.summary["val_" + name] = metric_value

        print(f"vali_results: {val_result}")
        return val_result

    def _train(self):
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

                # Log info
                progress_bar.update(self.batch_size)
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    refresh=True,
                )

                if self.wandb:
                    wandb.log({"loss": loss.item()}, step=i)

                self.model_optim.step()

    def resume(self, expeiment_checkpoint_path) -> Dict[str, float]:
        pass

    def run(self, seed=42) -> Dict[str, float]:
        self._setup_run(seed)
        print(f"run : {self.current_run} in seed: {seed}")
        print(
            f"model parameters: {sum([p.nelement() for p in self.model.parameters()])}"
        )
        epoch_time = time.time()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self._train()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # evaluate on vali set
            self._evaluate()

            self._save(seed=seed)

        return self._test()

    def runs(self, seeds: List[int] = [42, 234, 123, 345, 821]):
        results = []
        for i, seed in enumerate(seeds):
            self.current_run = i
            result = self.run(seed=seed)
            results.append(result)

            for name, metric_value in result.items():
                wandb.run.summary["test_" + name] = metric_value

        df = pd.DataFrame(results)
        self.metric_mean_std = df.agg(["mean", "std"]).T
        print(
            self.metric_mean_std.apply(
                lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
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
