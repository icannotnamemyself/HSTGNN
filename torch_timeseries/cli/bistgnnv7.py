import fire
from torch_timeseries.experiments.bistgnnv7_experiment import BiSTGNNv7Experiment

if __name__ == "__main__":
    fire.Fire(BiSTGNNv7Experiment)