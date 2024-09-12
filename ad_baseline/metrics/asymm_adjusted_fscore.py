import numpy as np
from typing import Optional
from .base import PointAdjustedMetric


class AsymmetricallyAdjustedFScore(PointAdjustedMetric):
    def __init__(self,
                 adjust_width: Optional[int] = 2,
                 beta: Optional[float] = 1.,
                 ):
        super(AsymmetricallyAdjustedFScore, self).__init__(beta=beta)
        self._adjust_width = int(adjust_width)

    @classmethod
    def name(cls) -> str:
        return 'asymmetrically_adjusted_fscore'

    def adjust(self,
               y_true: np.ndarray,
               y_pred: np.ndarray,
               ) -> np.ndarray:
        y_adj = y_pred.copy()
        y_null = np.zeros_like(y_pred)
        w = self._adjust_width
        y_patch = np.ones(2 * self._adjust_width + 1)
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
                    y_null[j] = 1.0

            if anomaly_continue:
                y_adj[i] = 1.0
                y_null[i] = 1.0

        y_extended = np.concatenate([np.zeros(w),
                                     y_pred,
                                     np.zeros(w)], axis=0)
        for i in range(y_pred.shape[0]):
            if y_pred[i]:
                j = i + w
                y_extended[j-w:j+w+1] = y_patch

        y_adj = np.maximum(y_extended[w:-w], y_adj)
        return y_adj


