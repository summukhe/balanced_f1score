import numpy as np
import matplotlib as mpl
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from ad_baseline.utility.score import check_binary_data
from ad_baseline.utility.range import rle_binary


def plot_anomaly(x: np.ndarray,
                 normal_color: Optional[str] = 'gray',
                 anomaly_color: Optional[str] = 'red',
                 normal_marker: Optional[str] = 'o',
                 anomaly_marker: Optional[str] = 'x',
                 marker_size: Optional[float] = 5,
                 line_color: Optional[str] = 'black',
                 line_width: Optional[float] = 0.25,
                 y_offset: Optional[float] = 0.,
                 ax: Optional[mpl.axes.Axes] = None,
                 plot_width: Optional[float] = 30.,
                 figsize: Optional[Tuple[float, float]] = (20, 2),
                 **kwargs):
    local_instance = False
    x = np.array(x)
    if not check_binary_data(x, relax_diversity=True):
        from collections import Counter
        raise ValueError(f"Error invalid binary string of length ({len(x)}) and values ({Counter(x)})!")
    fig = None
    if ax is None:
        local_instance = True
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    x_ = np.linspace(0, 1, len(x)) * plot_width
    y = np.repeat(y_offset, len(x))

    ax.axhline(y_offset,
               xmin=-1.,
               xmax=plot_width + 1,
               color=line_color,
               linewidth=line_width)
    normal_index_selection = (x == 0.)
    anomaly_index_selection = (x == 1.)
    ax.scatter(x_[normal_index_selection],
               y[normal_index_selection],
               c=normal_color,
               marker=normal_marker,
               s=marker_size)

    ax.scatter(x_[anomaly_index_selection],
               y[anomaly_index_selection],
               c=anomaly_color,
               marker=anomaly_marker,
               s=marker_size)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
    return fig, ax if local_instance else ax


def plot_range_anomaly(x: np.ndarray,
                       anomaly_color: Optional[str] = 'red',
                       anomaly_width: Optional[float] = 2.5,
                       line_color: Optional[str] = 'black',
                       line_width: Optional[float] = 0.25,
                       y_offset: Optional[float] = 0.0,
                       ax: Optional[mpl.axes.Axes] = None,
                       figsize: Optional[Tuple[float, float]] = (20, 2),
                       **kwargs
                       ):
    local_instance = False
    x = np.array(x)
    if not check_binary_data(x, relax_diversity=True):
        from collections import Counter
        raise ValueError(f"Error invalid binary string of length ({len(x)}) and values ({Counter(x)})!")
    anomaly_ranges = rle_binary(x)
    fig = None
    if ax is None:
        local_instance = True
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axhline(y_offset,
               xmin=0,
               xmax=len(x),
               color=line_color,
               linewidth=line_width)
    for s, e in anomaly_ranges:
        ax.plot([s, e],
                [y_offset, y_offset],
                color=anomaly_color,
                linestyle='-',
                linewidth=anomaly_width)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
    return fig, ax if local_instance else ax

