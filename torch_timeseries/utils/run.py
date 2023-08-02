# Import the W&B Python Library and log into W&B
from dataclasses import dataclass
from typing import Type, Tuple, List
import wandb
from torch_timeseries.experiments.experiment import Experiment


@dataclass
class Config:
    device: str
    horizons: List[int]
    datasets: List[Tuple[str, int]]

    epochs: int = 100
    patience: int = 5


def run(exp: Type[Experiment], config: Config, project:str, name:str, ):
    for dataset_type, windows in config.datasets:
        for horizon in config.horizons:
            exp = exp(
                epochs=config.epochs,
                patience=config.patience,
                windows=windows,
                horizon=horizon,
                dataset_type=dataset_type,
                device=config.device,
            )
            exp.config_wandb("BiSTGNN", "baseline")
            exp.runs()
            wandb.finish()
