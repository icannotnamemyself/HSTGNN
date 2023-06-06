import fire
from torch_timeseries.experiments.informer_experiment import InformerExperiment

if __name__ == "__main__":
    fire.Fire(InformerExperiment)