import fire
from torch_timeseries.experiments.tntcn_experiment import TNTCNExperiment

if __name__ == "__main__":
    fire.Fire(TNTCNExperiment)