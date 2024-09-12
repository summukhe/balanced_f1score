import os
import matplotlib.pyplot as plt
from ad_baseline.utility import plot_anomaly
from ad_baseline.generator import generate_gt_sequence, restricted_label_sequence_with_score


if __name__ == "__main__":
    x = generate_gt_sequence(n_instance=2,
                             anomaly_size=(2, 5),
                             anomaly_gap=(10, 25),
                             seed=1)
    y, _, _ = restricted_label_sequence_with_score(ref_x=x,
                                                   true_positives=1,
                                                   false_positives=3,
                                                   anomaly_size=(2, 5),
                                                   false_detection_size=(1, 2),
                                                   true_positive_min_coverage=0.5,
                                                   seed=121,
                                                   max_attempts=50)
    fig, axs = plt.subplots(2, 1, figsize=(10, 1))
    plot_anomaly(x, marker_size=50, ax=axs[0])
    plot_anomaly(y, marker_size=50, anomaly_color='orange', ax=axs[1])
    image_filename = os.path.join(os.path.dirname(__file__), 'images', 'anomaly_plot.png')
    plt.savefig(image_filename, dpi=200)
    plt.show()

