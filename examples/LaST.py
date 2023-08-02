

# Import the W&B Python Library and log into W&B
from dataclasses import dataclass
from typing import Tuple, List
import wandb
wandb.login()

from torch_timeseries.experiments.last import LaSTExperiment


@dataclass
class Config:
    device : str 
    horizons : List[int] 
    datasets : List[Tuple[str, int]]

    epochs :int = 100
    patience :int = 5

def run(config: Config):
    for dataset_type, windows in config.datasets:
        for horizon in config.horizons:
            exp = LaSTExperiment(
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
    
    

def main():
    horizons = [3,6,12,24]
    config = Config(
        device="cuda:0",
        horizons=horizons,
        datasets=[
            ("ETTm1", 384),
            ("ETTm2", 384),
            ("ETTh1", 384),
            ("ETTh2", 384),
            ("ExchangeRate", 96),
            ("Weather", 168),
        ]
    )

    run(config)
    
    

if __name__ == "__main__":
    main()