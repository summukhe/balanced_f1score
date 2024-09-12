import warnings
import numpy as np
from ad_baseline.utility import rle_binary, range_width
from typing import Tuple, Optional, List, Union


ArrayTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray]


def generate_gt_sequence(n_instance: int,
                         anomaly_size: Tuple[int, int],
                         anomaly_gap: Tuple[int, int],
                         seed: Optional[int] = 1,
                         rng: Optional[np.random.Generator] = None,
                         ):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    regular_segment_widths = np.array([rng.integers(*anomaly_gap) for i in range(n_instance + 1)])
    anomaly_segment_width = np.array([rng.integers(*anomaly_size) for i in range(n_instance)])

    data = [np.zeros(regular_segment_widths[0])]
    for i in range(n_instance):
        data.append(np.ones(anomaly_segment_width[i]))
        data.append(np.zeros(regular_segment_widths[i + 1]))
    x = np.concatenate(data, axis=0)
    return x


def select_from_ranges(n_select: int,
                       width_range: Tuple[int, int],
                       range_list: List[Tuple[int, int]],
                       rng: np.random.Generator,
                       max_sequence_length: int,
                       allow_reselect: Optional[bool] = True,
                       minimum_coverage: Optional[float] = 0.0,
                       max_attempt: Optional[int] = 5,
                       ):
    minimum_coverage = min(max(minimum_coverage, 0.), 1.)
    total_w = np.sum([e - s for s, e in range_list])
    segment_start = np.cumsum([0] + [e - s for s, e in range_list])
    marks, marks_within, seg_ids, attempts, cov = list(), list(), set(), 0, 0
    max_cov = 0.
    sel_marks = list()
    if n_select == 0:
        return sel_marks

    while ((len(seg_ids) < n_select) or (cov < minimum_coverage)) and (attempts < max_attempt):
        if not allow_reselect:
            rem_select = n_select
            seg_selected = []
            while rem_select >= len(range_list):
                seg_selected = seg_selected + list(range(len(range_list)))
                rem_select -= len(range_list)
            seg_selected = seg_selected + list(rng.choice(len(range_list), rem_select, replace=False))
            pos = []
            for s in seg_selected:
                r_s, r_e = range_list[s]
                adjust = segment_start[s]
                pos.append(rng.integers(0, r_e - r_s) + adjust)
            pos = np.array(pos)
        else:
            pos = np.sort(rng.choice(total_w, n_select, replace=False))
        w = rng.integers(*width_range, n_select)
        marks = list()
        marks_within = list()
        seg_ids, seg_order = set(), list()
        for i, p in enumerate(pos):
            s_start, s_end = p, p + w[i]
            seg_id = np.where(segment_start > s_start)[0][0] - 1
            seg_ids.add(seg_id)
            seg_order.append(seg_id)
            e_p = min(s_end, segment_start[seg_id + 1])
            if e_p - s_start >= 1:
                right_shift = range_list[seg_id][0]
                left_shift = segment_start[seg_id]
                marks.append((s_start + right_shift - left_shift,
                              min(max_sequence_length, s_end + right_shift - left_shift)))
                marks_within.append((s_start + right_shift - left_shift, e_p + right_shift - left_shift))
        attempts += 1
        if len(marks_within) == 0:
            cov = 0.
        else:
            cov = np.min([range_width(m_range)/range_width(range_list[s_id])
                          for m_range, s_id in zip(marks_within, seg_order)])
        if cov > max_cov:
            max_cov = cov
            sel_marks = marks.copy()
    return sel_marks


def restricted_label_sequence_with_score(ref_x: np.ndarray,  # true signal
                                         true_positives: int,
                                         false_positives: int,
                                         anomaly_size: Tuple[int, int],
                                         false_detection_size: Optional[Tuple[int, int]] = None,
                                         true_positive_min_coverage: Optional[float] = 0.3,
                                         r_range: Optional[Tuple[float, float]] = (0, 0.4),
                                         a_range: Optional[Tuple[float, float]] = (0.6, 1.0),
                                         ma_window: Optional[int] = 1,
                                         threshold: Optional[float] = None,
                                         max_attempts: Optional[int] = 5,
                                         seed: Optional[int] = 1,
                                         rng: Optional[np.random.Generator] = None,
                                         ) -> ArrayTriplet:
    a_start, a_end = a_range
    r_start, r_end = r_range
    assert (a_start < a_end) and (a_start >= 0) and (a_end <= 1), \
        f"Error: invalid anomaly range [{a_start:.2f}, {a_end:.2f}]"
    assert (r_start < r_end) and (r_start >= 0) and (a_end <= 1), \
        f"Error: invalid regular signal range [{r_start:.2f}, {r_end:.2f}]"
    assert (r_start + r_end) <= (a_start + a_end), \
        f'Error: anomaly signal should be strictly higher than regular signal!'

    if false_detection_size is None:
        false_detection_size = anomaly_size

    if rng is None:
        rng = np.random.default_rng(seed)

    a_ranges = rle_binary(ref_x)
    r_ranges, last_value = [], 0
    for s, e in a_ranges:
        if last_value < s:
            r_ranges.append((last_value, s))
        last_value = e
    if (len(a_ranges) > 0) and (a_ranges[-1][1] < len(ref_x)):
        r_ranges.append((a_ranges[-1][1], len(ref_x)))

    # n_instance = true_positives + false_positives
    false_detections = false_positives
    true_detections = true_positives

    f_marks = select_from_ranges(n_select=false_detections,
                                 width_range=false_detection_size,
                                 range_list=r_ranges,
                                 rng=rng,
                                 allow_reselect=True,
                                 max_sequence_length=len(ref_x))

    t_marks = select_from_ranges(n_select=true_detections,
                                 width_range=anomaly_size,
                                 range_list=a_ranges,
                                 allow_reselect=False,
                                 max_sequence_length=len(ref_x),
                                 rng=rng,
                                 minimum_coverage=true_positive_min_coverage,
                                 max_attempt=max_attempts)
    x_ = np.zeros(len(ref_x))
    for s, e in f_marks + t_marks:
        x_[s:e] = 1.

    x_r = rng.random(len(ref_x)) * (r_end - r_start) + r_start
    x_a = rng.random(len(ref_x)) * (a_end - a_start) + a_start
    s_ = (1 - x_) * x_r + x_ * x_a

    if ma_window > 1:
        s_ = np.convolve(s_, np.ones(ma_window) / ma_window, 'same')

    if (threshold is None) or (threshold > 1) or (threshold < 0.):
        f = np.sum([e - s for s, e in a_ranges])/len(s_)
        threshold = np.quantile(s_, q=1. - f)
    y_ = (s_ > threshold).astype(float)
    return y_, s_, x_
