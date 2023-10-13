

# Import the W&B Python Library and log into W&B
import wandb

from torch_timeseries.utils.run import Config, run
wandb.login()

from torch_timeseries.experiments.film_experiment import FiLMExperiment


def main():
    horizons = [3,6,12,24]
    config = Config(
        device="cuda:4",
        horizons=horizons,
        datasets=[
            # ("ETTm1", 384),
            # ("ETTm2", 384),
            # ("ETTh1", 384),
            # ("ETTh2", 384),
            # ("ExchangeRate", 96),
            # ("SolarEnergy", 168),  #OOMqnq
            # ("Weather", 168),
            # ("METR_LA", 288), # OOM
            # ("PEMS_BAY", 288), # OOM
        ]
    )

    run(FiLMExperiment, config, "BiSTGNN", "baseline")



if __name__ == "__main__":
    main()