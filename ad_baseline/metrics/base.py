import numpy as np
from typing import Optional
from abc import ABCMeta, abstractmethod
from ad_baseline.utility import check_label_compatibility, binary_fbeta_score


class AnomalyMetric:
    __metaclass__ = ABCMeta

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplemented

    @abstractmethod
    def score_(self,
               y_true: np.ndarray,
               y_pred: np.ndarray,
               **kwargs) -> float:
        raise NotImplemented

    def __call__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 **kwargs) -> float:
        return self.score_(y_true=y_true,
                           y_pred=y_pred,
                           **kwargs)

    @classmethod
    def from_config(cls, **config):
        return cls(**config)


class PointAdjustedMetric(AnomalyMetric):
    def __init__(self,
                 beta: Optional[float] = 1.):
        super(PointAdjustedMetric, self).__init__()
        self._beta = beta

    @abstractmethod
    def adjust(self,
               y_true: np.ndarray,
               y_pred: np.ndarray,
               ) -> np.ndarray:
        raise NotImplemented

    def score_(self,
               y_true: np.ndarray,
               y_pred: np.ndarray,
               **kwargs) -> float:
        y_true, y_pred = check_label_compatibility(y_true, y_pred)
        y_adj = self.adjust(y_true=y_true,
                            y_pred=y_pred)
        return binary_fbeta_score(y_true=y_true,
                                  y_pred=y_adj,
                                  beta=self._beta)


