

# Import the W&B Python Library and log into W&B
import wandb
wandb.login()
from torch_timeseries.utils.run import Config, run

from torch_timeseries.experiments.nlinear_experiment import NLinearExperiment


def main():
    horizons = [3,6,12,24]
    config = Config(
        device="cuda:2",
        horizons=horizons,
        datasets=[
            # ("METR_LA", 288),
            # ("PEMS_BAY", 288),
            # ("ETTm1", 384),
            # ("ETTm2", 384),
            # ("ETTh1", 384),
            # ("ETTh2", 384),
            # ("ExchangeRate", 96),
            # ("Weather", 168),
            ("SolarEnergy", 168), 
        ]
    )

    run(NLinearExperiment, config, "BiSTGNN", "baseline")



if __name__ == "__main__":
    main()