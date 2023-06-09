from math import ceil
import pytest
import torch
from torch_timeseries.nn.metric import R2, Corr, TrendAcc, compute_r2, compute_corr
from torch_timeseries.utils.timefeatures import time_features
from sklearn.metrics import r2_score, classification_report
import numpy as np


def test_corr():
    corr = Corr()
    preds = torch.randn(3, 64, 128, 3)
    trues = torch.randn(3, 64, 128, 3)
    for i in range(3):
        pred = preds[i, :, :, :]
        true = trues[i, :, :, :]
        corr.update(pred, true)

    result = corr.compute()
    corr1 = compute_corr(preds.reshape(-1, 128, 3), trues.reshape(-1, 128, 3))

    assert round(float(result), 4) == round(float(corr1), 4)

    corr = Corr()
    preds = torch.randn(3, 64, 128)
    trues = torch.randn(3, 64, 128)
    for i in range(3):
        pred = preds[i, :, :]
        true = trues[i, :, :]
        corr.update(pred, true)

    result = corr.compute()
    corr1 = compute_corr(preds.reshape(-1, 128), trues.reshape(-1, 128))

    assert round(float(result), 4) == round(float(corr1), 4)


def test_r2():
    # single step prediction
    r2 = R2(128, multioutput="uniform_average")
    preds = torch.randn(2, 64, 128)
    trues = torch.randn(2, 64, 128)
    for pred, true in zip(preds, trues):
        r2.update(pred, true)

    result = r2.compute()
    r21 = float(compute_r2(trues.reshape(-1, 128), preds.reshape(-1, 128)))

    assert round(float(result), 4) == round(r21, 4)
    # multi step prediction
    r2 = R2(128, num_nodes=3, multioutput="uniform_average")
    preds = torch.randn(2, 64, 128, 3)
    trues = torch.randn(2, 64, 128, 3)
    for pred, true in zip(preds, trues):
        r2.update(pred, true)

    result = r2.compute()
    r21 = float(compute_r2(trues.reshape(-1, 128, 3), preds.reshape(-1, 128, 3)))
    assert round(float(result), 4) == round(r21, 4)

    r2 = R2(128, multioutput="variance_weighted")
    preds = torch.randn(2, 64, 128)
    trues = torch.randn(2, 64, 128)
    for pred, true in zip(preds, trues):
        r2.update(pred, true)

    result = r2.compute()
    r21 = float(
        compute_r2(trues.reshape(-1, 128), preds.reshape(-1, 128), "variance_weighted")
    )

    assert round(float(result), 4) == round(r21, 4)
    # multi step prediction
    r2 = R2(128, num_nodes=3, multioutput="variance_weighted")
    preds = torch.randn(2, 64, 128, 3)
    trues = torch.randn(2, 64, 128, 3)
    for pred, true in zip(preds, trues):
        r2.update(pred, true)

    result = r2.compute()
    r21 = float(
        compute_r2(
            trues.reshape(-1, 128, 3), preds.reshape(-1, 128, 3), "variance_weighted"
        )
    )
    assert round(float(result), 4) == round(r21, 4)


def test_trend_acc():
    tacc = TrendAcc()
    for i in range(3):
        xt = torch.randn(64, 8)
        pred = torch.randn(64, 8)
        true = pred
        tacc.update(pred, true, xt)

    result = tacc.compute()
    assert round(float(result)) == 1
