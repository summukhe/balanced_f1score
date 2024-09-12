"""
Reference:
----------
Xu H, Chen W, Zhao N, et al.
Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications.
In: Proceedings of the 2018 world wide web conference.
International world wide web conferences steering committee, republic and canton of Geneva. CHE,
WWW ’18
"""

import numpy as np
from typing import Optional
from .base import PointAdjustedMetric


class PointAdjustedFScore(PointAdjustedMetric):
    def __init__(self,
                 beta: Optional[float] = 1.,
                 ):
        super(PointAdjustedFScore, self).__init__(beta=beta)

    @classmethod
    def name(cls) -> str:
        return 'point_adjusted_fscore'

    def adjust(self,
               y_true: np.ndarray,
               y_pred: np.ndarray
               ) -> np.ndarray:
        y_adj = y_pred.copy()
        anomaly_start, anomaly_continue = None, False
        for i in range(y_true.shape[0]):
            if not y_true[i]:
                anomaly_start = None
                anomaly_continue = False
            elif anomaly_start is None:
                anomaly_start = i

            if (anomaly_start is not None) and y_pred[i]:
                anomaly_continue = True
                for j in range(anomaly_start, i):
                    y_adj[j] = 1.0

            if anomaly_continue:
                y_adj[i] = 1.0
        return y_adj

