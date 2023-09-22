import os
from torch_timeseries.utils.run import run, print_id,Config
from torch_timeseries.experiments.bistgnnv2_experiment import BiSTGNNv2Experiment


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    

def main():
    horizons = [3,6,12,24]
    config = Config(
        device="cuda:6",
        horizons=horizons,
        datasets=[
            # ("ETTm1", 384),
            ("ETTm2", 384),
            # ("ETTh1", 384),
            # ("ETTh2", 384),
            ("ExchangeRate", 96),
            ("Weather", 168),
            # ("METR_LA", 288),
            # ("PEMS_BAY", 288),
        ],
        
        model_paramaeters={
            'graph_build_type': 'adaptive',
            'gcn_type':'han',
            'model_type': 'TNTCN_hetero_han',
        }
    )

    run(BiSTGNNv2Experiment, config, "BiSTGNN", "BiSTGNN_hetero_han")
if __name__ == "__main__":
    main()