import os
from torch_timeseries.utils.run import run, print_id,Config
from torch_timeseries.experiments.bistgnnv1_experiment import BiSTGNNv1Experiment


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    

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
            # ("Weather", 168),
            # ("METR_LA", 288),
            ("PEMS_BAY", 288),
        ],
        
        model_paramaeters={
            'graph_build_type': 'adaptive',
            'gcn_type':'fastgcn6',
            'model_type': 'TNTCN_hetero_fastgcn_6',
        }
    )

    run(BiSTGNNv1Experiment, config, "BiSTGNN", "BiSTGNN_hetero_fastgcn6")
if __name__ == "__main__":
    main()