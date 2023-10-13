import fire
from torch_timeseries.experiments.bistgnnv3_experiment import BiSTGNNv3Experiment

if __name__ == "__main__":
    fire.Fire(BiSTGNNv3Experiment)