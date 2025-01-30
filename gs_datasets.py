import numpy as np

from data_read.data_arff import create_2d4c, create_2d10c, create_2d20c, create_3spiral, create_aggregation, create_compound, create_elly_2d10c13s, create_s1, create_s2
from data_read.data_csv import create_unbalance
from data_read.data_sklearn import create_data1, create_data2, create_data3, create_data4, create_data5
from data_read.data_uci import create_ecoli, create_glass, create_wdbc, create_wine, create_sonar, create_ionosphere, create_statlog, create_yeast


def load_all_data():
    n_samples = 1000
    datasets = [
        ("D1", create_data1(n_samples)),
        ("D2", create_data2(n_samples)),
        ("D3", create_data3(n_samples)),
        ("D4", create_data4(n_samples)),
        ("D5", create_data5(n_samples)),
        ("3spiral",create_3spiral()  ),
        ("unbalance", create_unbalance()),
        ("2d4c",create_2d4c()  ),
        ("2d10c",create_2d10c()  ),
        ("2d20c",create_2d20c()  ),
        ("aggregation",create_aggregation()  ),
        ("compound",create_compound()  ),
        ("elly2d10c13s",create_elly_2d10c13s()  ),
        ("s1",create_s1()  ),
        ("s2",create_s2()),
        ("ecoli", create_ecoli()),
        ("glass", create_glass()),
        ("yeast", create_yeast()),
        ("statlog", create_statlog()),
        ("wdbc", create_wdbc()),
        ("wine", create_wine()),
        ("sonar", create_sonar()),
        ("ionosphere", create_ionosphere()),
    ]

    return datasets


def load_data_simple():
    n_samples = 1000
    datasets = [
        ("D1", create_data1(n_samples)),
        ("D2", create_data2(n_samples)),
        ("D3", create_data3(n_samples)),
    ]

    return datasets


def load_data_overlap():
    datasets = [
        ("s1", create_s1()),
        ("s2", create_s2()),
    ]
    return datasets


def load_data_imbalance():
    datasets = [
        ("2d4c", create_2d4c()),
        ("2d10c", create_2d10c()),
        ("2d20c", create_2d20c()),
        ("unbalance", create_unbalance()),
    ]

    return datasets
def load_data_overlap_and_imbalance():
    n_samples = 1000
    datasets = [
        ("aggregation", create_aggregation()),
        ("compound", create_compound()),
        ("elly2d10c13s", create_elly_2d10c13s()),
    ]

    return datasets



def load_data_nonconvex():
    n_samples = 1000
    datasets = [
        ("D4", create_data4(n_samples)),
        ("D5", create_data5(n_samples)),
        ("3spiral",create_3spiral()  ),
    ]

    return datasets

def load_data_hd():
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

    return datasets








# OLD LOAD BY TYPE
# def load_arff_data():
#     datasets = [
#         ("2d4c",create_2d4c()  ),
#         ("2d10c",create_2d10c()  ),
#         ("2d20c",create_2d20c()  ),
#         ("3spiral",create_3spiral()  ),
#         ("aggregation",create_aggregation()  ),
#         ("compound",create_compound()  ),
#         ("elly2d10c13s",create_elly_2d10c13s()  ),
#         ("s1",create_s1()  ),
#         ("s2",create_s2()),
#     ]
#
#     return datasets
#
# def load_uci_data():
#     datasets = [
#         ("ecoli", create_ecoli()),
#         ("glass", create_glass()),
#         ("yeast", create_yeast()),
#         ("statlog", create_statlog()),
#         ("wdbc", create_wdbc()),
#         ("wine", create_wine()),
#         ("sonar", create_sonar()),
#         ("ionosphere", create_ionosphere()),
#     ]
#
#     return datasets
#
# def load_sklearn_data():
#     n_samples = 1000
#     datasets = [
#         ("D1", create_data1(n_samples)),
#         ("D2", create_data2(n_samples)),
#         ("D3", create_data3(n_samples)),
#         ("D4", create_data4(n_samples)),
#         ("D5", create_data5(n_samples)),
#     ]
#
#     return datasets
#
# def load_csv_data():
#     datasets = [
#         ("unbalance", create_unbalance()),
#     ]
#
#     return datasets


def load_sklearn_data_3_multiple_dimensions():
    n_samples_list = [100, 500, 1000, 5000, 10000, 50000]
    n_features_list = [2, 5, 10, 50]
    datasets = []
    for n_samples in n_samples_list:
        for n_features in n_features_list:
            datasets.append((f"multiple_dimensions_D3_{n_samples}_{n_features}", create_data3(n_samples, n_features)))

    return datasets


if __name__ == "__main__":
    dataset = load_all_data()
    for name, (X, y) in dataset:
        data = np.hstack((X, y.reshape(-1, 1)))
        np.savetxt(f"./data/{name}.csv", data, delimiter=",")