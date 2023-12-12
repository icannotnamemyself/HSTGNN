import fire
from torch_timeseries.experiments.hstgnn_experimentv3 import HSTGNNv3Experiment

if __name__ == "__main__":
    fire.Fire(HSTGNNv3Experiment)