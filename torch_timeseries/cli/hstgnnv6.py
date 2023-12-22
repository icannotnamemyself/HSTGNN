import fire
from torch_timeseries.experiments.hstgnn_experimentv6 import HSTGNNv6Experiment

if __name__ == "__main__":
    fire.Fire(HSTGNNv6Experiment)