import pandas as pd


def create_unbalance():
    data = pd.read_csv('./data/unbalance.csv', header=None)
    temp_data = data.to_numpy()
    data = (temp_data[:, :-1], temp_data[:, -1])
    return data
