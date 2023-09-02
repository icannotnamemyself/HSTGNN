

# Import the W&B Python Library and log into W&B
import os
import wandb
wandb.login()

from torch_timeseries.experiments.tsmixer import TSMixerExperiment

from torch_timeseries.utils.run import run, Config


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    

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
        ]
    )

    run(TSMixerExperiment, config, "BiSTGNN", "baseline")

if __name__ == "__main__":
    main()

