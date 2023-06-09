
import pytest
import os
from torch_timeseries.datasets import SolarEnergy


def test_solar_energy():
    dataset = SolarEnergy(root='./data')
    assert os.path.exists("./data/solar_AL/solar_AL.csv") is True
    assert dataset.data.shape[0] == dataset.length and dataset.data.shape[1] == dataset.num_features
