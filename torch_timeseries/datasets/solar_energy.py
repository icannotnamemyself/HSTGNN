import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import torch
from typing import Callable, List, Optional
import os
import resource
import numpy as np
from.dataset import Dataset, TimeSeriesDataset


class SolarEnergy(TimeSeriesDataset):
    """The raw data is in http://www.nrel.gov/grid/solar-power-data.html : 
    It contains the solar power production records in the year of 2006, 
    which is sampled every 10 minutes from 137 PV plants in Alabama State.
    """
    name: str = 'solar_AL'
    num_features: int = 137
    sample_rate: int  # in munites
    length : int = 52560
    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz",
            self.dir,
            filename="solar_AL.txt.gz",
            md5="41ef7fdc958c2ca3fac9cd06d6227073",
        )

    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.dir, 'solar_AL.txt')
        self.data = np.loadtxt(self.file_name, delimiter=',')
        return self.data
