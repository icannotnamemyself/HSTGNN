import os
from torch_timeseries.utils.run import run, Config
from torch_timeseries.experiments.tntcn_experiment import TNTCNExperiment


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    

def main():
    horizons = [3,6,12,24]
    config = Config(
        # batch_size=128,
        device="cuda:3",
        horizons=horizons,
        datasets=[
            ("ETTm1", 384),
            ("ETTm2", 384),
            ("ETTh1", 384),
            ("ETTh2", 384),
            ("ExchangeRate", 96),
            ("Weather", 168),
        ],
        
        model_paramaeters={
            'graph_build_type': 'weighted_random_clip',
            'output_module': 'mlp',
            'model_type': 'TNMLP'
        }
    )

    run(TNTCNExperiment, config, "BiSTGNN", "BiSTGNN_mlp")

if __name__ == "__main__":
    main()