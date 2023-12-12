import fire
from torch_timeseries.experiments.hstgnn_experimentv4 import HSTGNNv4Experiment

if __name__ == "__main__":
    fire.Fire(HSTGNNv4Experiment)