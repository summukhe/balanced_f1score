"""
Kim S, Choi K, Choi H, et al.
Towards a rigorous evaluation of time-series anomaly detection.
In: Thirty-sixth AAAI conference on artificial intelligence, AAAI 2022,
Thirty-fourth conference on innovative applications of artificial intelligence,
IAAI 2022,
"""

import numpy as np
from typing import Optional
from .base import PointAdjustedMetric


class CoverageAdjustedFScore(PointAdjustedMetric):
    def __init__(self,
                 coverage: Optional[float] = 0.05,
                 erase_tp: Optional[bool] = False,
                 beta: Optional[float] = 1.,
                 ):
        super(CoverageAdjustedFScore, self).__init__(beta=beta)
        self._coverage = coverage
        self._erase_tp = erase_tp

    @classmethod
    def name(cls) -> str:
        return 'coverage_restricted_point_adjusted_fscore'

    def adjust(self,
               y_true: np.ndarray,
               y_pred: np.ndarray,
               ) -> np.ndarray:
        y_adj = y_pred.copy()
        anomaly_start, anomaly_continue = None, False
        anomaly_width, anomaly_count = 0, 0

        for i in range(y_true.shape[0]):
            if y_true[i] == 0.0:
                if (anomaly_start is not None) and \
                        (not anomaly_continue) and \
                        (anomaly_count > 0) and \
                        (self._erase_tp is True):
                    for j in range(anomaly_start, i):
                        y_adj[j] = 0.0
                anomaly_start = None
                anomaly_continue = False
                anomaly_count = 0
                anomaly_width = 0
            elif anomaly_start is None:
                anomaly_start = i
                while (i + anomaly_width < y_true.shape[0]) and (y_true[i + anomaly_width] == 1.0):
                    anomaly_width += 1

            if y_true[i] and y_pred[i]:
                anomaly_count += 1

            if y_true[i] == 1:
                if (anomaly_count * 1.0 / max(1, anomaly_width)) >= self._coverage:
                    anomaly_continue = True
                    for j in range(anomaly_start, i):
                        y_adj[j] = 1.0
            if anomaly_continue:
                y_adj[i] = 1.0

        return y_adj


