
import pytest
import os
from torch_timeseries.datasets import Traffic,TrafficV2


def test_traffic():
    traffic = Traffic(root='./data')
    df = traffic.raw_df()
    assert os.path.exists("./data/traffic/raw/traffic.txt.gz") is True
    assert df.shape[0] == 17544 and df.shape[1] == 862



def test_trafficv2():
    traffic = TrafficV2(root='./data')
    assert os.path.exists("./data/traffic/raw/traffic.txt.gz") is True
    assert traffic.data.shape[0] == 17544 and traffic.data.shape[1] == 862


