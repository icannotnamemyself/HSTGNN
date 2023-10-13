

# Import the W&B Python Library and log into W&B
from dataclasses import dataclass
from typing import Tuple, List
import wandb
wandb.login()
from torch_timeseries.utils.run import run, Config

from torch_timeseries.experiments.mtgnn_experiment import MTGNNExperiment


def main():
    horizons = [3,6,12,24]
    config = Config(
        device="cuda:4",
        horizons=horizons,
        datasets=[
            # ("ExchangeRate", 96),
            # ("ETTm1", 384),
            # ("ETTm2", 384),
            # ("ETTh1", 384),
            # ("ETTh2", 384),
            ("Weather", 168),
            ("SolarEnergy", 168), 
            ("METR_LA", 288),
            ("PEMS_BAY", 288),
        ]
    )

    run(MTGNNExperiment, config, "BiSTGNN", "baseline")

if __name__ == "__main__":
    main()