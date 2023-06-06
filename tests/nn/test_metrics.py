from math import ceil
import pytest
import torch
from torch_timeseries.nn.metric import R2, Corr, TrendAcc
from torch_timeseries.utils.timefeatures import time_features


def test_corr():
    corr = Corr()
    for i in range(3):
        pred = torch.randn(64, 128, 3)
        true = pred
        corr.update(pred, true)

    result = corr.compute()
    assert round(result) == 1

    corr = Corr()
    for i in range(3):
        pred = torch.randn(64, 3)
        true = pred
        corr.update(pred, true)

    result = corr.compute()
    assert round(result) == 1


def test_r2():
    r2 = R2(num_outputs=3)
    for i in range(3):
        pred = torch.randn(64, 128, 3)
        true = pred
        r2.update(pred, true)

    result = r2.compute()
    assert round(float(result)) == 1


def test_trend_acc():
    tacc = TrendAcc()
    for i in range(3):
        xt = torch.randn(64, 8)
        pred = torch.randn(64, 8)
        true = pred
        tacc.update(pred, true, xt)

    result = tacc.compute()
    assert round(float(result)) == 1
