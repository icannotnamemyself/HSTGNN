import datetime
import json
import os
import random
import time
import hashlib
from torch.cuda.amp import autocast, GradScaler
from prettytable import PrettyTable

####
from typing import Dict, List, Type, Union
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection, MeanAbsoluteError
from tqdm import tqdm
import wandb
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceRandomSplitter, SequenceSplitter
from torch_timeseries.datasets.dataloader import (
    ChunkSequenceTimefeatureDataLoader,
    DDPChunkSequenceTimefeatureDataLoader,
)
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.models.Informer import Informer
from torch.nn import MSELoss, L1Loss
from omegaconf import OmegaConf

from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from torch.nn import DataParallel
from dataclasses import asdict, dataclass

from torch_timeseries.nn.metric import R2, Corr, TrendAcc, compute_corr, compute_r2
from torch_timeseries.utils.early_stopping import EarlyStopping
import json
import codecs
from .experiment import Experiment



@dataclass
class SingleStepForecast(Experiment):
    horizon: int = 3
    
    def _init_metrics(self):
        self.metrics = MetricCollection(
            metrics={
                "r2": R2(self.dataset.num_features, multioutput="uniform_average"),
                "r2_weighted": R2(
                    self.dataset.num_features, multioutput="variance_weighted"
                ),
                "mse": MeanSquaredError(),
                "corr": Corr(),
                "mae": MeanAbsoluteError(),
            }
        )
        
        self.metrics.to(self.device)
