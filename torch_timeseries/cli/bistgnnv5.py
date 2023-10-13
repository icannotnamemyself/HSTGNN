import fire
from torch_timeseries.experiments.bistgnnv5_experiment import BiSTGNNv5Experiment

if __name__ == "__main__":
    fire.Fire(BiSTGNNv5Experiment)