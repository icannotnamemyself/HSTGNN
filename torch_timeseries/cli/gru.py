import fire
from torch_timeseries.experiments.gru_experiment import GRUExperiment

if __name__ == "__main__":
    fire.Fire(GRUExperiment)