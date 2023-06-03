
import pytest
import os
from torch_timeseries.datasets import ETTh1, ETTh2, ETTm1, ETTm2

def test_etth1():
    etth1 = ETTh1(root='./data')
    assert os.path.exists("./data/ETTh1/raw/ETTh1.csv") is True
    assert etth1.data.shape[0] == etth1.num_features and etth1.data.shape[1] == etth1.length


def test_etth1():
    etth2 = ETTh2(root='./data')
    assert os.path.exists("./data/ETTh2/ETTh2.csv") is True
    assert etth2.data.shape[0] == etth2.length and etth2.data.shape[1] == etth2.num_features


def test_ettm1():
    ettm1 = ETTm1(root='./data')
    assert os.path.exists("./data/ETTm1/ETTm1.csv") is True
    assert ettm1.data.shape[0] == ettm1.length and ettm1.data.shape[1] == ettm1.num_features


def test_ettm2():
    ett = ETTm2(root='./data')
    assert os.path.exists("./data/ETTm2/ETTm2.csv") is True
    assert ett.data.shape[0] == ett.length and ett.data.shape[1] == ett.num_features
