import fire
from torch_timeseries.experiments.astgcn_experiment import ASTGCNExperiment

if __name__ == "__main__":
    fire.Fire(ASTGCNExperiment)