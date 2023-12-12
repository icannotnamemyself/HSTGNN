import fire
from torch_timeseries.experiments.hstgnn_experimentv2 import HSTGNNv2Experiment

if __name__ == "__main__":
    fire.Fire(HSTGNNv2Experiment)