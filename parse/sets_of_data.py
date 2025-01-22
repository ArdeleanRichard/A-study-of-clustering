import numpy as np
from parse.data_uci import create_ecoli, create_glass, create_wdbc, create_wine, create_sonar, create_ionosphere, create_statlog, create_yeast


def create_setHD():
    data1 = create_ecoli()
    data2 = create_glass()
    data3 = create_yeast()
    data4 = create_statlog()
    data5 = create_wdbc()
    data6 = create_wine()
    data7 = create_sonar()
    data8 = create_ionosphere()

    datasets = [
        data1,
        data2,
        data3,
        data4,
        data5,
        data6,
        data7,
        data8,
    ]

    return datasets