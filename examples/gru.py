import os
from torch_timeseries.utils.run import run, Config

from torch_timeseries.experiments.gru_experiment import GRUExperiment


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    

def main():
    horizons = [3,6,12,24]
    config = Config(
        device="cuda:1",
        horizons=horizons,
        datasets=[
            ("ExchangeRate", 94),
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

    # run(TNTCNExperiment, config, "BiSTGNN", "baseline")
    run(GRUExperiment, config, "", "")

if __name__ == "__main__":
    main()