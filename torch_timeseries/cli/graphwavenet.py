import fire
from torch_timeseries.experiments.graphwavenet_experiment import GraphWavenetExperiment

if __name__ == "__main__":
    fire.Fire(GraphWavenetExperiment)