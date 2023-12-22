import fire
from torch_timeseries.experiments.hstgnn_experimentv5 import HSTGNNv5Experiment

if __name__ == "__main__":
    fire.Fire(HSTGNNv5Experiment)