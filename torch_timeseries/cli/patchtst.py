import fire
from torch_timeseries.experiments.patchtst_experiment import PatchTSTExperiment

if __name__ == "__main__":
    fire.Fire(PatchTSTExperiment)
