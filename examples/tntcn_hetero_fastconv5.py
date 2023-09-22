import os
from torch_timeseries.utils.run import run, print_id,Config
from torch_timeseries.experiments.tntcn_experiment import TNTCNExperiment


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    

def main():
    horizons = [3,6,12,24]
    config = Config(
        device="cuda:5",
        horizons=horizons,
        datasets=[
            # ("ETTm1", 384),
            # ("ETTm2", 384),
            # ("ETTh1", 384),
            # ("ETTh2", 384),
            # ("ExchangeRate", 96),
            # ("Weather", 168),
            ("METR_LA", 288),
            ("PEMS_BAY", 288),
        ],
        
        model_paramaeters={
            'graph_build_type': 'nt_full_connected',
            'gcn_type': 'heterofastgcn5',
            'model_type': 'TNTCN_hetero_fastgcn_5',
            'gcn_eps': 2, # adapative
        }
    )

    run(TNTCNExperiment, config, "BiSTGNN", "BiSTGNN_hetero_fastgcn5")
if __name__ == "__main__":
    main()