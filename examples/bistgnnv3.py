import os
from torch_timeseries.utils.run import run, print_id,Config,run_seed
from torch_timeseries.experiments.bistgnnv3_experiment import BiSTGNNv3Experiment


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    
# [42,233,666,19971203,19980224]

def main():
    horizons = [12,24,6]
    config = Config(
        device="cuda:1",
        horizons=horizons,
        datasets=[
            # ("ETTm1", 384),
            # ("ETTm2", 384),
            # ("ETTh1", 384),
            # ("ETTh2", 384),
            # ("ExchangeRate", 96),
            # ("Weather", 168),
            ("METR_LA", 288),
            # ("PEMS_BAY", 288),
            # ("SolarEnergy", 168), 
        ],
        batch_size=32, # 32 for METR_LA Weather PEMS_BAY  ExchangeRate , 128 for others
        model_paramaeters={
            'graph_build_type': 'adaptive',
            'gcn_type':'han',
        }
    )
    

    run(BiSTGNNv3Experiment, config, "BiSTGNN", "BiSTGNN")
    # run(BiSTGNNv2Experiment, config, "BiSTGNN", "BiSTGNN_hetero_han")
    
    # run_seed(BiSTGNNv2Experiment, config,19980224)
if __name__ == "__main__":
    main()