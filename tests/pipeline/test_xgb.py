
from tabnanny import verbose
from typing import Dict, Tuple
import numpy as np
# from sklearn.preprocessing import MaxAbsScaler
import xgboost as xgb
from torch.utils.data import DataLoader , Subset, RandomSampler
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch_timeseries.data.scaler import MaxAbsScaler
from torch_timeseries.datasets.electricity import ElectricityV2
from torch_timeseries.datasets.splitter import SequenceRandomSplitter
from torch_timeseries.datasets.wrapper import MultiStepFlattenWrapper
from torch_timeseries.pipelines.xgb import XGBoostPipeline


def test_xgboost(dummy_dataset):
    seed = 42
    params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    "max_depth":4,
    "n_jobs":16,
    'tree_method': 'gpu_hist',
    'gpu_id': 2,
    'n_estimators': 128,
    "multi_strategy": 'multi_output_tree'  , # one_output_per_tree multi_output_tree
    "seed": seed  , # one_output_per_tree multi_output_tree
    'eval_metric': 'mse'  # 设置验证指标为平均绝对误差（MAE）
    }

    # Train the XGBoost model
    # model = xgb.train(params, dtrain)

    electricity = ElectricityV2('./data')
    dataset = electricity
    
    model = xgb.XGBRegressor(**params)

    dataset = MultiStepFlattenWrapper(dataset, steps=1, window=168, horizon=3)
    srs = SequenceRandomSplitter(seed)
    p = XGBoostPipeline(model, srs)
    
    p.train(dataset)
    
    model.get_booster()
    
    