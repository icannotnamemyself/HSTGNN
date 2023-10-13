import fire
from torch_timeseries.experiments.dcrnn_experiment import DCRNNExperiment

if __name__ == "__main__":
    fire.Fire(DCRNNExperiment)