
import pytest
import os
from torch_timeseries.datasets import ExchangeRate


def test_exchange_rate():
    exchange_rate = ExchangeRate(root='./data')
    df = exchange_rate.raw_df()
    assert os.path.exists("./data/exchange_rate/raw/exchange_rate.txt.gz") is True
    print(df)
    assert df.shape[0] == 7587 and df.shape[1] == 8


