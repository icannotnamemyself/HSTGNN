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


def run(exp_type: Type[Experiment], config: Config, project:str="", name:str=""):
    
    for dataset_type, windows in config.datasets:
        for horizon in config.horizons:
            exp = exp_type(
                epochs=config.epochs,
                patience=config.patience,
                windows=windows,
                horizon=horizon,
                dataset_type=dataset_type,
                device=config.device,
            )
            if project != "" and name != "":
                exp.config_wandb(project, name)
            exp.runs()
            wandb.finish()



