import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    pd.set_option('use_inf_as_na', True)
    csv_file = os.path.join(os.path.dirname(__file__), 'output', 'scenario_1.csv')
    df = pd.read_csv(csv_file, index_col=0)
    anchor_column = 'anomaly_coverage'
    df = df[(df.false_positives > 5)]
    df[anchor_column] = df[anchor_column].apply(lambda x: np.round(x, 1))

    pdf = pd.melt(df[[anchor_column, 'pa', 'kpa', 'fpa']],
                  id_vars=anchor_column,
                  value_vars=['pa', 'kpa', 'fpa'])

    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data=pdf,
                 x=anchor_column,
                 y='value',
                 hue='variable',
                 color=".5",
                 ax=ax,
                 )
    plt.show()

