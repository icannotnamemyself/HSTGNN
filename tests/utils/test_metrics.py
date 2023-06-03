from typing import Dict, Tuple
import numpy as np
# from sklearn.preprocessing import MaxAbsScaler
import xgboost as xgb
from torch.utils.data import DataLoader , Subset, RandomSampler
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch_timeseries.utils.metrics import rse, rae


def test_rse():
    y_true = np.random.random((3,4))
    y_pred = np.random.random((3,4))
    rse_ = rse(y_true, y_pred)
    


def test_rae():
    y_true = np.random.random((3,4))
    y_pred = np.random.random((3,4))
    rae_ = rae(y_true, y_pred)
