
import pytest
import os
from torch_timeseries.datasets import ExchangeRate


def test_exchange_rate():
    dataset = ExchangeRate(root='./data')
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features
