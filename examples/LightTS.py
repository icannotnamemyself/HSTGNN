from torch_timeseries.utils.run import run, Config
from torch_timeseries.experiments.lightts_experiment import LightTSExperiment


    

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
            ("METR_LA", 288),
            ("PEMS_BAY", 288),

        ]
    )

    run(LightTSExperiment, config)

if __name__ == "__main__":
    main()