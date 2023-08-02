import fire
from torch_timeseries.experiments.multiple_step_forcast import MTGNNExperiment

if __name__ == "__main__":
    fire.Fire(MTGNNExperiment)
