import fire
from torch_timeseries.experiments.STSGCN_experiment import STSGCNExperiment

if __name__ == "__main__":
    fire.Fire(STSGCNExperiment)