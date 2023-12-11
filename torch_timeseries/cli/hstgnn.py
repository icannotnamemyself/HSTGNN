import fire
from torch_timeseries.experiments.hstgnn_experiment import HSTGNNExperiment

if __name__ == "__main__":
    fire.Fire(HSTGNNExperiment)