import fire
from torch_timeseries.experiments.hstgnn_experimentv7 import HSTGNNv7Experiment

if __name__ == "__main__":
    fire.Fire(HSTGNNv7Experiment)