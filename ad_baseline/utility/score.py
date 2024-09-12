import numpy as np
from typing import Optional
from sklearn.metrics import fbeta_score


def check_binary_data(x: np.ndarray,
                      relax_dim: Optional[bool] = False,
                      relax_diversity: Optional[bool] = False,
                      relax_validity: Optional[bool] = False,
                      ):
    x = np.asarray(x)
    valid_dim = x.ndim == 1
    valid_value_diversity = len(np.unique(x)) == 2
    valid_binary_values = (np.sum(x == 0) + np.sum(x == 1) == x.shape[0])
    return ((valid_dim or relax_dim) and
            (valid_value_diversity or relax_diversity) and
            (valid_binary_values or relax_validity))


def check_label_compatibility(y_true: np.ndarray,
                              y_pred: np.ndarray):
    y_true, y_pred = np.asarray(y_true).astype(float), np.asarray(y_pred).astype(float)
    if not check_binary_data(y_true):
        raise ValueError(f"Error: invalid y_true data for fbeta score!")

    if not check_binary_data(y_pred, relax_diversity=True):
        raise ValueError(f"Error: invalid y_pred data for fbeta score!")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Error: mismatch in y_true and y_pred size ({y_true.shape} != {y_pred.shape})")
    return y_true, y_pred


def binary_fbeta_score(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       beta: Optional[float] = 1,
                       pos_label: Optional[float] = 1.,
                       sample_weight: Optional[np.ndarray] = None,
                       ):
    y_true, y_pred = check_label_compatibility(y_true, y_pred)

    return fbeta_score(y_true,
                       y_pred,
                       beta=beta,
                       average='binary',
                       pos_label=pos_label,
                       sample_weight=sample_weight)


