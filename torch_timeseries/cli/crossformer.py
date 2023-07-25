import fire
from torch_timeseries.experiments.crossformer_experiment import CrossformerExperiment

if __name__ == "__main__":
    fire.Fire(CrossformerExperiment)