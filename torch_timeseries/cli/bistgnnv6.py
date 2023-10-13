import fire
from torch_timeseries.experiments.bistgnnv6_experiment import BiSTGNNv6Experiment

if __name__ == "__main__":
    fire.Fire(BiSTGNNv6Experiment)