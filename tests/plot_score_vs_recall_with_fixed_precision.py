import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    pd.set_option('use_inf_as_na', True)
    csv_file = os.path.join(os.path.dirname(__file__), 'output', 'scenario_1.csv')
    df = pd.read_csv(csv_file, index_col=0)

    df['coverage'] = '< 20%'
    df['coverage'][(df.anomaly_coverage > 0.2) & (df.anomaly_coverage < 0.3)] = '(20%, 30%)'
    df['coverage'][(df.anomaly_coverage > 0.3)] = '> 30%'

    df['tpr'] = df.true_positives / (df.true_positives + df.false_positives)
    anchor_column = 'event_recall'

    df['precision'] = '< 25%'
    df['precision'][(df.tpr > 0.25) & (df.tpr < 0.75) ] = '(25%, 75%)'
    df['precision'][df.tpr >= 0.75] = '> 75%'

    df[anchor_column] = df[anchor_column].apply(lambda x: np.round(x, 1))

    name_map = dict(pa='$F_{1}PA$',
                    kpa='$F_{1}KPA$',
                    fpa='$F_{1}BA^{\\ast}$',
                    fscore='$F_{1}P$')

    pdf = pd.melt(df[[anchor_column, 'precision', 'coverage', 'fscore', 'pa', 'kpa', 'fpa']],
                  id_vars=['coverage', 'precision', anchor_column],
                  value_vars=['fscore', 'pa', 'kpa', 'fpa'])
    pdf['variable'] = pdf['variable'].apply(lambda x: name_map.get(x, x))

    g = sns.FacetGrid(pdf[pdf.precision == '(25%, 75%)'],
                      col='coverage',
                      margin_titles=True,
                      col_order=['< 20%', '(20%, 30%)', '> 30%'],
                      )
    g.map_dataframe(sns.lineplot,
                    anchor_column,
                    'value',
                    'variable',
                    color=".3",
                    errorbar=('sd', 0.25),
                    style='variable',
                    )
    g.add_legend()
    g.set_titles(col_template="{col_name}", fontweight='bold', size=18)
    g.set_xlabels('$Recall_{E}$', fontsize=16)
    g.set_ylabels('$F_{1} score$', fontsize=16)
    image_file = os.path.join(os.path.dirname(__file__),
                              'images',
                              'recall_sensitivity.pdf')
    plt.savefig(image_file)
    plt.show()

