import fire
import torch
import torch.nn as nn

from torch_timeseries.experiments.soft_mlp_expert import SoftMLPExpertExperiment
from torch_timeseries.nn.embedding import *



if __name__ == "__main__":
    fire.Fire(SoftMLPExpertExperiment)


