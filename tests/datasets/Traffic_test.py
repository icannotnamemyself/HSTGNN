
import pytest
import os
from torch_timeseries.datasets import Traffic


def test_traffic():
    traffic = Traffic('./data')
    assert os.path.exists("./data/traffic/raw/traffic.txt.gz") is True


