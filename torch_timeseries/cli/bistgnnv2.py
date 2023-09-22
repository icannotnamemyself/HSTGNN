import fire
from torch_timeseries.experiments.bistgnnv2_experiment import BiSTGNNv2Experiment

if __name__ == "__main__":
    fire.Fire(BiSTGNNv2Experiment)