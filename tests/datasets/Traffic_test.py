
import pytest
import os
from torch_timeseries.datasets import Traffic


def test_traffic():
    traffic = Traffic(root='./data')
    df = traffic.raw_df()
    assert os.path.exists("./data/traffic/raw/traffic.txt.gz") is True
    assert df.shape[0] == 17543 and df.shape[1] == 862


