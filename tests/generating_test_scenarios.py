import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from ad_baseline.generator import separation_metric
from ad_baseline.generator import (generate_gt_sequence,
                                   restricted_label_sequence_with_score)
from ad_baseline.utility import (rle_binary,
                                 check_overlap,
                                 identify_range_overlaps,
                                 range_width,
                                 overlapped_length)
from ad_baseline.metrics import (AsymmetricallyAdjustedFScore,
                                 SimpleFScore,
                                 PointAdjustedFScore,
                                 CoverageAdjustedFScore)


def run_scenario_1(n_iteration: int = 1500,
                   min_anomaly: int = 1,
                   max_anomaly: int = 15,
                   anomaly_width: Tuple[int, int] = (50, 150),
                   anomaly_gap: Tuple[int, int] = (500, 2000),
                   true_detection_width: Tuple[int, int] = (5, 15),
                   false_detection_width: Tuple[int, int] = (1, 5),
                   min_true_positive: float = 0.5,
                   max_true_positive: float = 1.0,
                   min_false_positives: int = 0,
                   max_false_positives: int = 20,
                   ma_window: int = 5,
                   seed: int = 121,
                   distance_metric: str = 'hellinger',
                   ):
    """
    Change number of anomalies keeping coverage fixed!
    """
    rng = np.random.default_rng(seed=seed)
    anomaly_count, anomaly_fraction, sep_metric = list(), list(), list()
    pa_score, kpa_score, fpa_score, fsc_score = list(), list(), list(), list()
    tp_detect, fp_detect = list(), list()
    detection_coverage, event_recall = list(), list()

    fp_count = min_false_positives
    fp_increment_interval = n_iteration // max(1, max_false_positives - min_false_positives)

    adjust_width = range_width(anomaly_width) // 4

    pa = PointAdjustedFScore()
    kpa = CoverageAdjustedFScore(coverage=0.2)
    fpa = AsymmetricallyAdjustedFScore(adjust_width=adjust_width)
    fsc = SimpleFScore()
    d_step, d_warp = 0.0127, 0.5
    d = 0
    for iter_count in tqdm(range(n_iteration)):
        n_anomalies = rng.integers(min_anomaly, max_anomaly)
        x = generate_gt_sequence(n_anomalies,
                                 anomaly_size=anomaly_width,
                                 anomaly_gap=anomaly_gap,
                                 rng=rng)
        x_ranges = rle_binary(x)
        a_count = len(x_ranges)
        a_fraction = np.sum([e - s for s, e in x_ranges]) / len(x)

        d = (d + d_step) % d_warp
        r_max = 0.25 + d
        a_min = 0.75 - d
        tp_count = rng.integers(int(min_true_positive * n_anomalies),
                                int(max_true_positive * n_anomalies))
        if (iter_count > 0) and (iter_count % fp_increment_interval == 0):
            fp_count += 1
        z_detect, z_score, z = restricted_label_sequence_with_score(x,
                                                                    true_positives=tp_count,
                                                                    false_positives=fp_count,
                                                                    anomaly_size=true_detection_width,
                                                                    false_detection_size=false_detection_width,
                                                                    r_range=(0, r_max),
                                                                    a_range=(a_min, 1),
                                                                    threshold=0.5,
                                                                    ma_window=ma_window,
                                                                    rng=rng)
        detected_ranges = rle_binary(z_detect)

        tp_detected = len([s for s in detected_ranges if check_overlap(s, x_ranges)])
        fp_detected = len(detected_ranges) - tp_detected

        coverage = [overlapped_length(s, detected_ranges)/range_width(s) for s in x_ranges
                    if check_overlap(s, detected_ranges)]
        coverage = np.mean(coverage) if len(coverage) > 0 else 0

        marked_anom = []
        for s in detected_ranges:
            marked_anom += identify_range_overlaps(s, x_ranges)
        marked_anom = list(set(marked_anom))

        sep_metric.append(separation_metric(x, z_score, metric=distance_metric))
        anomaly_fraction.append(a_fraction)
        detection_coverage.append(coverage)
        event_recall.append(len(marked_anom)/max(1, a_count))
        anomaly_count.append(a_count)
        tp_detect.append(tp_detected)
        fp_detect.append(fp_detected)
        pa_score.append(pa(x, z_detect))
        kpa_score.append(kpa(x, z_detect))
        fpa_score.append(fpa(x, z_detect))
        fsc_score.append(fsc(x, z_detect))

    dataframe = pd.DataFrame(dict(anomaly_count=anomaly_count,
                                  anomaly_fraction=anomaly_fraction,
                                  signal_separation=sep_metric,
                                  true_positives=tp_detect,
                                  false_positives=fp_detect,
                                  event_recall=event_recall,
                                  anomaly_coverage=detection_coverage,
                                  fscore=fsc_score,
                                  pa=pa_score,
                                  kpa=kpa_score,
                                  fpa=fpa_score))
    return dataframe


if __name__ == "__main__":
    run_experiment = 0
    if run_experiment == 1:
        score_df = run_scenario_1(n_iteration=15000,
                                  min_anomaly=10,
                                  max_anomaly=24,
                                  anomaly_width=(50, 100),
                                  anomaly_gap=(200, 1000),
                                  true_detection_width=(10, 25),
                                  false_detection_width=(2, 5),
                                  ma_window=15,
                                  min_true_positive=0.0,
                                  max_true_positive=1.0)
        out_file = os.path.join(os.path.dirname(__file__),
                                'output',
                                'scenario_1.csv')
        score_df.to_csv(out_file)
    else:
        from ad_baseline.utility import plot_anomaly
        import matplotlib.pyplot as plt
        x = generate_gt_sequence(2,
                                 anomaly_size=(5, 10),
                                 anomaly_gap=(20, 30))
        y, scores, z_ = restricted_label_sequence_with_score(x,
                                                             true_positives=2,
                                                             false_positives=3,
                                                             anomaly_size=(2, 5),
                                                             false_detection_size=(2, 5),
                                                             a_range=(0.6, 1.0),
                                                             r_range=(0., 0.5),
                                                             threshold=0.6,
                                                             ma_window=2,
                                                             seed=None,
                                                             true_positive_min_coverage=0.1,
                                                             )
        metrics = [SimpleFScore(),
                   PointAdjustedFScore(),
                   AsymmetricallyAdjustedFScore(adjust_width=3),
                   CoverageAdjustedFScore(coverage=0.3)]
        print('s_score', separation_metric(x, scores))
        print('f_score', " ".join([f"{m(x, y):.4f}" for m in metrics]))
        print('rle : x', rle_binary(x))
        print('rle : z', rle_binary(z_))

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 2.5))
        plot_anomaly(x, marker_size=25, plot_width=30, anomaly_color='red', ax=axs[0])
        plot_anomaly(z_, marker_size=25, plot_width=30, anomaly_color='blue', y_offset=2.0, ax=axs[1])
        plot_anomaly(y, marker_size=25, plot_width=30, anomaly_color='orange', y_offset=-1.5, ax=axs[1])

        axs[1].fill_between(np.linspace(0, 30, len(scores)),
                            scores,
                            color='green',
                            linewidth=1.5,
                            alpha=0.5)
        plt.tight_layout()

        image_file = os.path.join(os.path.dirname(__file__), 'images', 'generator_anomaly.png')
        plt.savefig(image_file, dpi=300)
        plt.show()


