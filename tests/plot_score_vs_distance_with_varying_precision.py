import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    pd.set_option('use_inf_as_na', True)
    csv_file = os.path.join(os.path.dirname(__file__), 'output', 'scenario_1.csv')
    df = pd.read_csv(csv_file, index_col=0)

    df['tpr'] = df.true_positives / (df.true_positives + df.false_positives)

    df['precision'] = '< 25%'
    df['precision'][(df.tpr >= 0.25) & (df.tpr < 0.75)] = '(25%, 75%)'
    df['precision'][df.tpr >= 0.75] = '> 75%'

    df['recall'] = '< 25%'
    df['recall'][(df.event_recall > 0.25) & (df.event_recall < 0.75)] = '(25%, 75%)'
    df['recall'][df.event_recall >= 0.75] = '> 75%'

    anchor_column = 'signal_separation'
    df[anchor_column] = df[anchor_column].apply(lambda x: np.round(x, 2))

    name_map = dict(pa='$F_{1}PA$',
                    kpa='$F_{1}KPA$',
                    fpa='$F_{1}BA^{\\ast}$',
                    fscore='$F_{1}P$')

    pdf = pd.melt(df[[anchor_column, 'precision', 'recall', 'fscore', 'pa', 'kpa', 'fpa']],
                  id_vars=['precision', 'recall', anchor_column],
                  value_vars=['fscore', 'pa', 'kpa', 'fpa'])
    pdf['variable'] = pdf['variable'].apply(lambda x: name_map.get(x, x))

    pdf = pdf[pdf[anchor_column] < 0.5]
    g = sns.FacetGrid(pdf[pdf.recall == '(25%, 75%)'],
                      col='precision',
                      margin_titles=True,
                      col_order=['< 25%', '(25%, 75%)', '> 75%'],
                      )
    g.map_dataframe(sns.lineplot,
                    anchor_column,
                    'value',
                    'variable',
                    color=".3",
                    errorbar=None,
                    style='variable',
                    )
    g.add_legend()
    g.set_titles(col_template="{col_name}", fontweight='bold', size=18)
    g.set_xlabels('Separation', fontsize=16)
    g.set_ylabels('$F_{1} score$', fontsize=16)
    image_file = os.path.join(os.path.dirname(__file__),
                              'images',
                              'separation_sensitivity.pdf')
    plt.savefig(image_file)
    plt.show()

