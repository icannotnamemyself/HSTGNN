import fire
from torch_timeseries.experiments.bistgnnv8_experiment import BiSTGNNv8Experiment

if __name__ == "__main__":
    fire.Fire(BiSTGNNv8Experiment)