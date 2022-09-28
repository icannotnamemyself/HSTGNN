
import pytest
import os
from torch_timeseries.datasets import SolarEnergy


def test_solar_energy():
    solar_energy = SolarEnergy(root='./data')
    df = solar_energy.raw_df()
    assert os.path.exists("./data/solar_AL/raw/solar_AL.txt.gz") is True
    assert df.shape[0] == 52559 and df.shape[1] == 137


