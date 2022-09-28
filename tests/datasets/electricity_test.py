
import pytest
import os
from torch_timeseries.datasets import Electricity


def test_exchange_rate():
    electricity = Electricity(root='./data')
    df = electricity.raw_df()
    assert os.path.exists("./data/electricity/raw/electricity.txt.gz") is True
    assert df.shape[0] == 26304 and df.shape[1] == 321


