import numpy as np
from typing import Optional
from .base import PointAdjustedMetric


class SimpleFScore(PointAdjustedMetric):
    def __init__(self,
                 beta: Optional[float] = 1.,
                 ):
        super(SimpleFScore, self).__init__(beta=beta)

    @classmethod
    def name(cls) -> str:
        return 'simple_fscore'

    def adjust(self,
               y_true: np.ndarray,
               y_pred: np.ndarray
               ) -> np.ndarray:
        return y_pred

