import pandas as pd


def create_data9():
    data = pd.read_csv('./datasets/unbalance.csv', header=None)
    temp_data = data.to_numpy()
    data = (temp_data[:, :-1], temp_data[:, -1])
    return data


def create_data11():
    data = pd.read_csv('./datasets/s1_labeled.csv', header=None)
    temp_data = data.to_numpy()
    data = (temp_data[:, :-1], temp_data[:, -1])
    return data