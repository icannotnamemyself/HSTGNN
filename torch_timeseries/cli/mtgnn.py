import fire
from torch_timeseries.experiments.mtgnn_experiment import MTGNNExperiment

if __name__ == "__main__":
    fire.Fire(MTGNNExperiment)
