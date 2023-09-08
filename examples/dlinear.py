

# Import the W&B Python Library and log into W&B
import wandb

from torch_timeseries.utils.run import Config, run
wandb.login()

from torch_timeseries.experiments.dlinear_experiment import DLinearExperiment

def main():
    horizons = [3,6,12,24]
    config = Config(
        device="cuda:2",
        horizons=horizons,
        datasets=[
            ("ETTm1", 384),
            ("ETTm2", 384),
            ("ETTh1", 384),
            ("ETTh2", 384),
            ("ExchangeRate", 96),
            ("Weather", 168),
            ("METR_LA", 288),
            ("PEMS_BAY", 288),
        ]
    )

    run(DLinearExperiment, config, "BiSTGNN", "baseline")



if __name__ == "__main__":
    main()