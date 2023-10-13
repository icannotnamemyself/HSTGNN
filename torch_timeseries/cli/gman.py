import fire
from torch_timeseries.experiments.GMAN_experiment import GMANExperiment

if __name__ == "__main__":
    fire.Fire(GMANExperiment)