import fire
from torch_timeseries.experiments.timesnet_experiment import TimesNetExperiment

if __name__ == "__main__":
    fire.Fire(TimesNetExperiment)
