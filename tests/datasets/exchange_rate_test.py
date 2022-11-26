
import pytest
import os
from torch_timeseries.datasets import ExchangeRate,ExchangeRateV2


def test_exchange_rate():
    exchange_rate = ExchangeRate(root='./data')
    df = exchange_rate.raw_df()
    assert os.path.exists("./data/exchange_rate/raw/exchange_rate.txt.gz") is True
    assert df.shape[0] == 7588 and df.shape[1] == 8


def test_exchange_ratev2():
    exchange_rate = ExchangeRateV2(root='./data')
    assert os.path.exists("./data/exchange_rate/raw/exchange_rate.txt.gz") is True
    assert exchange_rate.data.shape[0] == 7588 and exchange_rate.data.shape[1] == 8

