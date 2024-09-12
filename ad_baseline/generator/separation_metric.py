import numpy as np
from typing import Optional
from scipy.stats import gaussian_kde
from ad_baseline.utility import rle_binary, check_binary_data


def discrete_bhattacharyya_distance(p: np.ndarray,
                                    q: np.ndarray,
                                    eps: Optional[float] = 1e-10,
                                    ):
    p = np.array(p)
    q = np.array(q)
    if len(p) != len(q):
        raise ValueError(f"Error: incompatible discrete vectors!")

    if (np.sum(p < 0) > 0) or (np.sum(q < 0) > 0):
        raise ValueError(f"Error: invalid probability vector!")

    return -np.log(np.max(np.sum(p * q), eps))


def discrete_hellinger_distance(p: np.ndarray,
                                q: np.ndarray):
    p = np.array(p)
    q = np.array(q)
    if len(p) != len(q):
        raise ValueError(f"Error: incompatible discrete vectors!")

    if (np.sum(p < 0) > 0) or (np.sum(q < 0) > 0):
        raise ValueError(f"Error: invalid probability vector!")
    return 0.707107 * np.sqrt(np.sum(np.square(np.sqrt(p) - np.sqrt(q))))


def separation_metric(x_true: np.ndarray,
                      x_scores: np.ndarray,
                      metric: Optional[str] = 'hellinger',
                      bins: Optional[int] = 100,
                      eps: Optional[float] = 1e-10,
                      ):
    if not check_binary_data(x_true, relax_diversity=True):
        raise ValueError(f"Error: not a valid binary labels!")

    x_ranges = rle_binary(x_true)
    x_r = []
    x_a = []
    last_pos = 0
    if metric.lower() not in ('hellinger', 'bhattacharyya'):
        raise ValueError(f"Error: unknown metric {metric}!")

    for i, a_range in enumerate(x_ranges):
        s, e = a_range
        x_a.append(x_scores[s:e])
        if last_pos != s:
            x_r.append(x_scores[last_pos:s])
        last_pos = e
    if (len(x_ranges) > 0) and (x_ranges[-1][1] < len(x_true)):
        x_r.append(x_scores[last_pos:len(x_true)])
    if len(x_a) == 0:
        return 1
    x_a = np.concatenate(x_a)
    x_r = np.concatenate(x_r)
    min_a = np.min(x_a)
    max_r = np.max(x_r)
    x_r = x_r[(x_r > min_a) & (x_r < max_r)]
    if (len(x_r) > 0) and (len(x_a) > 0):
        g_r = gaussian_kde(x_r)
        q_r = gaussian_kde(x_a)
        x_bins = np.linspace(min_a, max_r, bins + 1)
        p_i = np.array([g_r.integrate_box_1d(float(x_bins[i]),
                                             float(x_bins[i+1]))
                        for i in range(bins)])
        q_i = np.array([q_r.integrate_box_1d(float(x_bins[i]),
                                             float(x_bins[i+1]))
                        for i in range(bins)])

        if metric.lower() == 'bhattacharyya':
            return discrete_bhattacharyya_distance(p_i, q_i, eps=eps)
        elif metric.lower() == 'hellinger':
            return discrete_hellinger_distance(p_i, q_i)
    else:
        return 1. if metric.lower() == 'hellinger' else 25.

