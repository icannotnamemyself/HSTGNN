
import pytest
import os
from torch_timeseries.datasets import SolarEnergy, SolarEnergyV2


def test_solar_energy():
    solar_energy = SolarEnergy(root='./data')
    df = solar_energy.raw_df()
    assert os.path.exists("./data/solar_AL/raw/solar_AL.txt.gz") is True
    assert df.shape[0] == 52560 and df.shape[1] == 137

def test_solar_energyv2():
    solar_energy = SolarEnergyV2(root='./data')
    assert os.path.exists("./data/solar_AL/raw/solar_AL.txt.gz") is True
    assert solar_energy.data.shape[0] == 52560 and solar_energy.data.shape[1] == 137


