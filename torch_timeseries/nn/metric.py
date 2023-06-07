from typing import Any

import torch
from torch import Tensor, tensor

from torchmetrics.metric import Metric
from torchmetrics import MetricCollection, R2Score, MeanSquaredError


class TrendAcc(Metric):
    """Metric for single step forcasting"""

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_trend_hit", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, xt: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        assert preds.shape == target.shape
        assert len(preds.shape) == 2
        batch_size = preds.shape[0]
        self.sum_trend_hit += ((preds - xt) * (target - xt) > 0).sum()
        self.total += batch_size * preds.shape[1]

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return self.sum_trend_hit / self.total


class R2(R2Score):
    """Metric for multi step forcasting"""

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        if len(preds.shape) <= 2:
            R2Score.update(self, preds, target)
        elif len(preds.shape) == 3:
            for i in range(preds.shape[1]):
                self.update(preds[:, i, :], target[:, i, :])


class Corr(Metric):
    """correlation for multivariate timeseries, Corr compute correlation for every columns/nodes and output the averaged result"""

    def __init__(self):
        super().__init__()
        self.add_state("y_pred", default=torch.Tensor(), dist_reduce_fx="cat")
        self.add_state("y_true", default=torch.Tensor(), dist_reduce_fx="cat")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.reshape(-1, y_true.shape[-1])
        self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)
        self.y_true = torch.cat([self.y_true, y_true], dim=0)

    def compute(self):
        sigma_p = self.y_pred.std(0)  # (node_num)
        sigma_g = self.y_true.std(0)  # (node_num)
        mean_p = self.y_pred.mean(0)  # (node_num)
        mean_g = self.y_true.mean(0)  # (node_num)
        index = sigma_g != 0
        sigma_p += 1e-7
        sigma_g += 1e-7
        (correlation) = ((self.y_pred - mean_p) * (self.y_true - mean_g)).mean(0) / (
            sigma_p * sigma_g
        )
        correlation = correlation[index].mean().item()
        return correlation
