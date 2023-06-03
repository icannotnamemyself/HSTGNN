from typing import Dict, Tuple
import numpy as np
# from sklearn.preprocessing import MaxAbsScaler
from torch_timeseries.utils.timefeatures import time_features
import pandas as pd


def test_time_features():
    # 生成日期序列
    dates = pd.date_range(start='2022-01-01', end='2022-01-02', freq='t')
    # 随机生成一些数据
    data = np.linspace(1, 10, len(dates))

    # 转换为 DataFrame
    df = pd.DataFrame({'date': dates, 'data': data})

    assert time_features(df, 0, 'm').shape == (len(df), 1)
    assert time_features(df, 0, 'w').shape == (len(df), 1)
    assert time_features(df, 0, 'd').shape == (len(df), 3)
    assert time_features(df, 0, 'b').shape == (len(df), 3)
    assert time_features(df, 0, 'h').shape == (len(df), 4)
    assert time_features(df, 0, 't').shape == (len(df), 5)

    assert time_features(df, 1, 'Q').shape == (len(df), 1)
    assert time_features(df, 1, 'B').shape == (len(df), 3)
    assert time_features(df, 1, 'M').shape == (len(df), 1)
    assert time_features(df, 1, 'T').shape == (len(df), 5)
    assert time_features(df, 1, 'W').shape == (len(df), 2)
    assert time_features(df, 1, 'D').shape == (len(df), 3)
    assert time_features(df, 1, 'H').shape == (len(df), 4)
    assert time_features(df, 1, 'S').shape == (len(df), 6)
