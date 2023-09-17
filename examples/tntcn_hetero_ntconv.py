import os
from torch_timeseries.utils.run import run, Config
from torch_timeseries.experiments.tntcn_experiment import TNTCNExperiment


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    

def main():
    horizons = [3,6,12,24]
    config = Config(
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
            'graph_build_type': 'full_connected',
            'gcn_type': 'heterofagcn',
            'model_type': 'TNTCN_hetero_nt1',
            'n_first': False,
            'gcn_eps': 1
        }
    )

    run(TNTCNExperiment, config, "BiSTGNN", "BiSTGNN_heter_nt1")

if __name__ == "__main__":
    main()