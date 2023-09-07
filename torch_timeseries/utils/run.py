# Import the W&B Python Library and log into W&B
from dataclasses import dataclass
from typing import Type, Tuple, List
import wandb
from torch_timeseries.experiments.experiment import Experiment
from dataclasses import dataclass, asdict, field

@dataclass
class Config:
    device: str
    horizons: List[int]
    datasets: List[Tuple[str, int]]
    batch_size : int = 128
    epochs: int = 100
    patience: int = 5
    
    model_paramaeters: dict = field(default_factory=lambda:{})


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
                **config.model_paramaeters
            )
            if project != "" and name != "":
                exp.config_wandb(project, name)
            exp.runs()
            wandb.finish()



