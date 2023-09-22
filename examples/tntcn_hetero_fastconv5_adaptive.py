import os
from torch_timeseries.utils.run import run, Config
from torch_timeseries.experiments.tntcn_experiment import TNTCNExperiment


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
            ("METR_LA", 288),
            ("PEMS_BAY", 288),
        ],
        
        model_paramaeters={
            'graph_build_type': 'nt_adaptive_graph',
            'gcn_type': 'heterofastgcn5',
            'model_type': 'TNTCN_hetero_fastgcn_5_adaptive',
            'gcn_eps': 1,
        }
    )

    run(TNTCNExperiment, config, "BiSTGNN", "BiSTGNN_hetero_fastgcn5_adaptive")

if __name__ == "__main__":
    main()