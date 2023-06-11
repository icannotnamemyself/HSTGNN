import fire
from torch_timeseries.experiments.informer_nots_experiment import InformerNotsExperiment

if __name__ == "__main__":
    fire.Fire(InformerNotsExperiment)