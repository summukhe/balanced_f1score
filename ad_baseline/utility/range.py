import numpy as np
from enum import Enum
from typing import Optional, List, Tuple, Union


def rle(x: np.ndarray):
    running_v, start_p = x[0], 0
    rle_seq = list()
    for i, v in enumerate(x):
        if v != running_v:
            rle_seq.append((running_v, start_p, i))
            start_p = i
            running_v = v
    rle_seq.append((running_v, start_p, x.shape[0]))
    return rle_seq


def rle_binary(x: np.ndarray):
    x = np.array(x).astype(float)
    marks = np.where(x == 1.0)[0]
    d_marks = np.diff(marks)
    splits = np.where(d_marks > 1)[0]
    start_p = 0
    range_seq = list()
    for s in splits:
        range_seq.append((marks[start_p], marks[s]+1))
        start_p = s + 1
    if len(marks) > 0:
        range_seq.append((marks[start_p], marks[-1]+1))
    return range_seq


def segment_reduction(y_true: np.ndarray,
                      y_pred: np.ndarray):
    y_pred_c = y_pred.copy()
    y_pred_c[(y_pred == 1.0) & (y_true != y_pred)] = 2.0  # FP = 2
    y_pred_c[(y_pred == 0.0) & (y_true != y_pred)] = 3.0  # FN = 3
    compressed = rle(y_pred_c)
    x_true, x_pred = list(), list()
    for v, s, e in compressed:
        if v == 0.0:
            x_true.append(0.)
            x_pred.append(0.)
        elif v == 1.0:
            x_true.append(1.)
            x_pred.append(1.)
        elif v == 2.0:
            x_true.append(0.)
            x_pred.append(1.)
        else:
            x_true.append(1.)
            x_pred.append(0.)
    return np.array(x_true).astype(float), np.array(x_pred).astype(float)


class RangeOverlap(Enum):
    subsumed = 'subsumed'
    included = 'included'
    front = 'front'
    back = 'back'
    no_intersection = 'no_intersection'


RangeType = Union[Tuple[int, int], Tuple[float, float]]


def is_valid_range(start: Union[int, float],
                   end: Union[int, float]) -> bool:
    return start < end


def range_width(x: RangeType) -> Union[int, float]:
    start, end = x
    return end - start


def range_intersection_type(ref_range: RangeType,
                            tgt_range: RangeType,
                            ) -> RangeOverlap:
    s_1, e_1 = ref_range
    s_2, e_2 = tgt_range
    if not is_valid_range(s_1, e_1):
        raise ValueError(f"Error: invalid range - 1 [{ref_range}]!")

    if not is_valid_range(s_2, e_2):
        raise ValueError(f"Error: invalid range - 2 [{tgt_range}]!")

    overlapped = RangeOverlap.no_intersection
    if (s_1 >= s_2) and (e_1 <= e_2):
        overlapped = RangeOverlap.subsumed
    elif (s_1 <= s_2) and (e_1 >= e_2):
        overlapped = RangeOverlap.included
    elif (s_1 >= s_2) and (s_1 < e_2):
        overlapped = RangeOverlap.back
    elif (e_1 > s_2) and (e_1 <= e_2):
        overlapped = RangeOverlap.front
    return overlapped


def check_overlap(ref_range: RangeType,
                  range_list: List[RangeType],
                  ) -> bool:
    for tgt_range in range_list:
        if range_intersection_type(ref_range, tgt_range) != RangeOverlap.no_intersection:
            return True
    return False


def find_range_overlaps(ref_range: RangeType,
                        range_list: List[RangeType],
                        ) -> List[RangeType]:
    overlaps = list()
    for tgt_range in range_list:
        if range_intersection_type(ref_range, tgt_range) != RangeOverlap.no_intersection:
            overlaps.append(tgt_range)
    return overlaps


def identify_range_overlaps(ref_range: RangeType,
                            range_list: List[RangeType],
                            ) -> List[int]:
    overlaps = list()
    for i, tgt_range in enumerate(range_list):
        if range_intersection_type(ref_range, tgt_range) != RangeOverlap.no_intersection:
            overlaps.append(i)
    return overlaps


def check_range_inclusion(pos: int,
                          range_list: List[RangeType]) -> bool:
    for s, e in range_list:
        if (pos >= s) and (pos < e):
            return True
    return False


def overlapped_length(ref_range: RangeType,
                      range_list: List[RangeType]
                      ) -> float:
    overlap = 0.
    for tgt_range in range_list:
        if range_intersection_type(ref_range, tgt_range) != RangeOverlap.no_intersection:
            overlap += min(ref_range[1], tgt_range[1]) - max(ref_range[0], tgt_range[0])
    return overlap


def check_non_intersecting_range(range_list: List[RangeType]) -> bool:
    range_list = sorted(range_list, key=lambda x: x[0])
    return all([range_list[i][1] <= range_list[i+1][0]
                for i in range(len(range_list) - 1)])


def get_intersecting_segment(ref_range: RangeType,
                             other_range: RangeType) -> Optional[RangeType]:
    if range_intersection_type(ref_range, other_range) != RangeOverlap.no_intersection:
        s_ref, e_ref = ref_range
        s_tgt, e_tgt = other_range
        return max(s_ref, s_tgt), min(e_ref, e_tgt)
    return None


def range_distance(x: Union[float, int, RangeType],
                   y: Union[RangeType, List[RangeType]]):
    if isinstance(y, list) and all([isinstance(y_, (list, tuple)) for y_ in y]):
        d_ = [range_distance(x, y_) for y_ in y]
        return np.nanmin(d_)

    s_t, e_t = y
    if not isinstance(x, (tuple, list, np.ndarray)):
        return 0 if (x >= s_t) and (x <= e_t) else min(abs(x - s_t), abs(x - e_t))
    else:
        d = 0
        o_type = range_intersection_type(tuple(x), y)
        if o_type == RangeOverlap.no_intersection:
            s, e = x
            d = min(abs(e - s_t), abs(s - e_t))
        return d


