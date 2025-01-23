"""
This file only exists for the plotting of data:
- Plots can be found in figs/datafigs/
"""

import matplotlib.pyplot as plt
import visualization.scatter_plot as sp
from parse.data_arff import create_2d4c, create_2d10c, create_2d20c, create_3spiral, create_aggregation, create_compound, create_elly_2d10c13s, create_s1, create_s2
from parse.data_csv import create_unbalance
from parse.data_sklearn import create_data1, create_data2, create_data3, create_data4, create_data5
from parse.data_uci import create_ecoli, create_glass, create_wdbc, create_wine, create_sonar, create_ionosphere, create_statlog, create_yeast


def plot_datasets(datasets, pca=None):
    for dataset in datasets:
        (title, (X, y)) = dataset
        if pca is None:
            sp.plot(title, X, y)
        else:
            sp.plot2D(title, X, y)
        plt.savefig("./figs/datafigs/"+title+".png")
        plt.close()


def draw_arff_data():
    datasets = [
        ("2d4c",create_2d4c()  ),
        ("2d10c",create_2d10c()  ),
        ("2d20c",create_2d20c()  ),
        ("3spiral",create_3spiral()  ),
        ("aggregation",create_aggregation()  ),
        ("compound",create_compound()  ),
        ("elly_2d10c13s",create_elly_2d10c13s()  ),
        ("s1",create_s1()  ),
        ("s2",create_s2()),
    ]
    plot_datasets(datasets)

def draw_uci_data():
    datasets = [
        ("ecoli", create_ecoli()),
        ("glass", create_glass()),
        ("yeast", create_yeast()),
        ("statlog", create_statlog()),
        ("wdbc", create_wdbc()),
        ("wine", create_wine()),
        ("sonar", create_sonar()),
        ("ionosphere", create_ionosphere()),
    ]

    plot_datasets(datasets, pca=True)

def draw_sklearn_data():
    n_samples = 1000
    datasets = [
        ("D1", create_data1(n_samples)),
        ("D2", create_data2(n_samples)),
        ("D3", create_data3(n_samples)),
        ("D4", create_data4(n_samples)),
        ("D5", create_data5(n_samples)),
    ]

    plot_datasets(datasets)

def draw_csv_data():
    datasets = [
        ("unbalance", create_unbalance()),
    ]

    plot_datasets(datasets)


if __name__ == "__main__":
    draw_arff_data()
    draw_uci_data()
    draw_sklearn_data()
    draw_csv_data()