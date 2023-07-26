import datetime
import json
import os
import random
import time
import hashlib
####
from typing import Dict, List, Type, Union
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection, MeanAbsoluteError
from tqdm import tqdm
import wandb
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceRandomSplitter, SequenceSplitter
from torch_timeseries.datasets.dataloader import (
    ChunkSequenceTimefeatureDataLoader,
    DDPChunkSequenceTimefeatureDataLoader,
)
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.nn.Informer import Informer
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf

from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from torch.nn import DataParallel
from dataclasses import asdict, dataclass

from torch_timeseries.nn.metric import R2, Corr, TrendAcc, compute_corr, compute_r2
from torch_timeseries.utils.early_stopping import EarlyStopping
import json
import codecs



@dataclass
class ResultRelatedSettings:
    dataset_type: str
    optm_type: str = "Adam"
    model_type: str = ""
    scaler_type: str = "StandarScaler"
    loss_func_type: str = "mse"
    batch_size: int = 32
    lr: float = 0.0003
    l2_weight_decay: float = 0.0005
    epochs: int = 1

    horizon: int = 3
    windows: int = 384
    pred_len: int = 1

    patience: int = 5
    max_grad_norm: float = 5.0


@dataclass
class Settings(ResultRelatedSettings):
    data_path: str = "./data"
    device: str = "cuda:0"
    num_worker: int = 20
    save_dir: str = "./results"
    experiment_label: str = str(int(time.time()))


class Experiment(Settings):
    def config_wandb(
        self,
        project: str,
        name: str,
    ):
        run = wandb.init(
            project=project,
            name=name,
            tags=[self.model_type, self.dataset_type, f"horizon-{self.horizon}", f"window-{self.windows}"],
        )
        wandb.config.update(asdict(self))
        self.wandb = True
        print(f"using wandb , running in config: {asdict(self)}")
        return self

    def _use_wandb(self):
        return hasattr(self, "wandb")

    
    def _run_print(self, *args, **kwargs):
        time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
        
        print(*args, **kwargs)
        if hasattr(self, "run_setuped") and getattr(self, "run_setuped") is True:
            with open(os.path.join(self.run_save_dir, 'output.log'), 'a+') as f:
                print(time, *args, flush=True, file=f)

    
    def config_wandb_verbose(
        self,
        project: str,
        name: str,
        tags: List[str],
        notes: str,
    ):
        run = wandb.init(
            project=project,
            name=name,
            notes=notes,
            tags=tags,
        )
        wandb.config.update(asdict(self))
        print(f"using wandb , running in config: {asdict(self)}")
        self.wandb = True
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
                "mae": MeanAbsoluteError(),
            }
        )
        
        self.metrics.to(self.device)


    def _run_identifier(self, seed) -> str:
        ident = asdict(self)

        keys_to_remove = [
            "data_path",
            "device",
            "num_worker",
            "save_dir",
            "experiment_label",
        ]
        for key in keys_to_remove:
            if key in ident:
                del ident[key]
        ident["seed"] = seed
        ident_md5 = hashlib.md5(
            json.dumps(ident, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return str(ident_md5)

    def _init_ddp_data_loader(self):
        self.dataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        self.dataloader = DDPChunkSequenceTimefeatureDataLoader(
            self.dataset,
            self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            scale_in_train=False,
            shuffle_train=True,
            # TODO: dataset specific freqency settings
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

    def _init_optimizer(self):
        self.model_optim = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optim, T_max=self.epochs
        )

    def _init_model(self):
        self.model = self._parse_type(self.model_type)().to(self.device)

    def _setup(self):
        # init data loader
        self._init_data_loader()

        # init metrics
        self._init_metrics()

        # init loss function based on given loss func type
        self._init_loss_func()

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
        self._init_model()

        self._init_optimizer()
        self.current_epoch = 0
        self.run_save_dir = os.path.join(
            self.save_dir,
            "runs",
            self.model_type,
            self.dataset_type,
            self._run_identifier(seed),
        )

        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "best_model.pth"
        )

        self.run_checkpoint_filepath = os.path.join(
            self.run_save_dir, "run_checkpoint.pth"
        )

        self.early_stopper = EarlyStopping(
            self.patience, verbose=True, path=self.best_checkpoint_filepath
        )
        
        
        self.run_setuped = True
        
        
        

    def _setup_dp_run(self, seed, device_ids, output_device):
        self._setup_run(seed)
        self.model = DataParallel(
            self.model, device_ids=device_ids, output_device=output_device
        ).to(self.device)

    def _parse_type(self, str_or_type: Union[Type, str]) -> Type:
        if isinstance(str_or_type, str):
            return eval(str_or_type)
        elif isinstance(str_or_type, type):
            return str_or_type
        else:
            raise RuntimeError(f"{str_or_type} should be string or type")

    def _setup_ddp_run(self, world_size=1):
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, find_unused_parameters=True
        )
        self._init_ddp_data_loader()
        self._init_metrics()

        self.current_epochs = 0
        self.current_run = 0

        self.setuped = True

    def _save(self, seed=0):
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

        # now only save the newest state
        torch.save(self.app_state, f"{self.checkpoint_filepath}")

    def _save_run_check_point(self, seed):
        # 检查目录是否存在
        if not os.path.exists(self.run_save_dir):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")

    def reproducible(self, seed):
        # for reproducibility
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determinstic = True

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
            # batch_x:  (B, T, N)
            # batch_y:  (B, T, Steps)
            # batch_x_date_enc:  (B, T, N)
            # batch_y_date_enc:  (B, T, Steps)
            
        # outputs:
            # pred: 
            # label:  (B,O,N)
        raise NotImplementedError()
    
    
    def _test(self) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        print("Testing .... ")
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
        if self._use_wandb():
            for name, metric_value in test_result.items():
                wandb.run.summary["test_" + name] = metric_value

        self._run_print(f"test_results: {test_result}")
        return test_result

    def _evaluate(self):
        self.model.eval()
        self.metrics.reset()

        # y_truths = []
        # y_preds = []

        print("Evaluating .... ")
        with tqdm(total=self.val_steps) as progress_bar:
            for batch_x, batch_y, batch_x_date_enc, batch_y_date_enc in self.val_loader:
                preds, truths = self._process_one_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                self.metrics.update(preds, truths)

                # y_truths.append(truths.detach().cpu().numpy())
                # y_preds.append(preds.detach().cpu().numpy())

                progress_bar.update(batch_x.shape[0])

        # y_truths = np.concatenate(y_truths, axis=0)
        # y_preds = np.concatenate(y_preds, axis=0)

        val_result = {
            name: float(metric.compute()) for name, metric in self.metrics.items()
        }
        # print(f"wjn corr: {compute_corr(y_truths, y_preds)}")
        # print(f"wjn r2: {compute_r2(y_truths, y_preds, aggr_mode='uniform_average')}")
        # print(
        #     f"wjn r2 weighted: {compute_r2(y_truths, y_preds, aggr_mode='variance_weighted')}"
        # )
        # compute_r2
        if self._use_wandb():
            for name, metric_value in val_result.items():
                wandb.run.summary["val_" + name] = metric_value

        self._run_print(f"vali_results: {val_result}")
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

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                progress_bar.update(self.batch_size)
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

                if self._use_wandb():
                    wandb.log({"loss": loss.item()}, step=i)

                self.model_optim.step()
                
    def _check_run_exist(self, seed: str):
        if not os.path.exists(self.run_save_dir):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.run_save_dir)
            print(f"Creating running results saving dir: '{self.run_save_dir}'.")
        else:
            print(f"result directory exists: {self.run_save_dir}")
        with codecs.open(os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

        exists = os.path.exists(self.run_checkpoint_filepath)
        return exists

    def _resume_run(self, seed):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        run_save_dir = os.path.join(
            self.save_dir,
            "runs",
            self.model_type,
            self.dataset_type,
            self._run_identifier(seed),
        )

        run_checkpoint_filepath = os.path.join(run_save_dir, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])
       
    def _load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_checkpoint_filepath, map_location=self.device))

    def run(self, seed=42) -> Dict[str, float]:
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")
        
        self.model_parameters_num = sum([p.nelement() for p in self.model.parameters()])
        self._run_print(
            f"model parameters: {self.model_parameters_num}"
        )
        if self._use_wandb():
            wandb.run.summary["parameters"] = self.model_parameters_num
        # for resumable reproducibility

        epoch_time = time.time()
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"loss no decreased for {self.patience} epochs,  early stopping ...."
                )
                break

            if self._use_wandb():
                wandb.run.summary["at_epoch"] = self.current_epoch
            # for resumable reproducibility
            self.reproducible(seed + self.current_epoch)
            self._train()

            self._run_print(
                "Epoch: {} cost time: {}".format(
                    self.current_epoch + 1, time.time() - epoch_time
                )
            )

            # evaluate on vali set
            result = self._evaluate()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(result[self.loss_func_type], model=self.model)
            
            self._save_run_check_point(seed)

            self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        self.run_setuped = False
        return best_test_result

    def dp_run(self, seed=42, device_ids: List[int] = [0, 2, 4, 6], output_device=0):
        self._setup_dp_run(seed, device_ids, output_device)
        print(f"run : {self.current_run} in seed: {seed}")
        print(
            f"model parameters: {sum([p.nelement() for p in self.model.parameters()])}"
        )
        epoch_time = time.time()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            if self._use_wandb():
                wandb.run.summary["at_epoch"] = epoch
            self._train()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # evaluate on vali set
            self._evaluate()

            self._save(seed=seed)

        return self._test()

    def runs(self, seeds: List[int] = [42,233,666,19971203,19980224]):
        results = []
        for i, seed in enumerate(seeds):
            self.current_run = i
            if self._use_wandb():
                wandb.run.summary["at_run"] = i
            result = self.run(seed=seed)
            results.append(result)

            if self._use_wandb():
                for name, metric_value in result.items():
                    wandb.run.summary["test_" + name] = metric_value

        df = pd.DataFrame(results)
        self.metric_mean_std = df.agg(["mean", "std"]).T
        print(
            self.metric_mean_std.apply(
                lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
            )
        )
        if self._use_wandb():
            for index, row in self.metric_mean_std.iterrows():
                wandb.run.summary[f"{index}_mean"] = row["mean"]
                wandb.run.summary[f"{index}_std"] = row["std"]
                wandb.run.summary[index] = f"{row['mean']:.4f}±{row['std']:.4f}"


def main():
    exp = Experiment(
        dataset_type="ETTm1",
        data_path="./data",
        optm_type="Adam",
        model_type="Informer",
        batch_size=32,
        device="cuda:3",
        windows=10,
        epochs=1,
        lr=0.001,
        pred_len=3,
        scaler_type="MaxAbsScaler",
    )

    # conf = OmegaConf.structured(exp)
    # print(OmegaConf.to_yaml(conf))

    exp._run_identifier()
    # exp = Experiment(settings)
    # exp.run()


if __name__ == "__main__":
    main()
