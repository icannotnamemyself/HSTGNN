
import pytest
import os
from torch_timeseries.datasets import Traffic


def test_traffic():
    dataset = Traffic(root='./data')
    assert os.path.exists("./data/traffic/raw/traffic.txt.gz") is True
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features

