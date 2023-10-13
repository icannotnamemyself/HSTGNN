import fire
from torch_timeseries.experiments.bistgnnv4_experiment import BiSTGNNv4Experiment

if __name__ == "__main__":
    fire.Fire(BiSTGNNv4Experiment)