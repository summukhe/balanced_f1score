import numpy as np
from typing import Optional
from .base import PointAdjustedMetric


class SymmetricallyAdjustedFScore(PointAdjustedMetric):
    def __init__(self,
                 adjust_width: Optional[int] = 2,
                 beta: Optional[float] = 1.,
                 ):
        super(SymmetricallyAdjustedFScore, self).__init__(beta=beta)
        self._adjust_width = int(adjust_width)

    @classmethod
    def name(cls) -> str:
        return 'symmetrically_adjusted_fscore'

    def adjust(self,
               y_true: np.ndarray,
               y_pred: np.ndarray,
               ) -> np.ndarray:
        w = self._adjust_width
        y_patch = np.ones(2 * self._adjust_width + 1)
        y_extended = np.concatenate([np.zeros(w),
                                     y_pred,
                                     np.zeros(w)], axis=0)
        for i in range(y_pred.shape[0]):
            if y_pred[i]:
                j = i + w
                y_extended[j-w:j+w+1] = y_patch
        y_adj = y_extended[w:-w]
        return y_adj


