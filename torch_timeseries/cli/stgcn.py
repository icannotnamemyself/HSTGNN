import fire
from torch_timeseries.experiments.stgcn_experiment import STGCNExperiment

if __name__ == "__main__":
    fire.Fire(STGCNExperiment)