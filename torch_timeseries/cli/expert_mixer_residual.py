import fire
import torch
import torch.nn as nn

from torch_timeseries.experiments.expert_inter_mixer import ExpertLayerWithResidualExperiment
from torch_timeseries.nn.embedding import *



if __name__ == "__main__":
    fire.Fire(ExpertLayerWithResidualExperiment)


