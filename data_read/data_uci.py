import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def read_uci(fetched_data):
    X = fetched_data.data.features.to_numpy()
    y = fetched_data.data.targets.to_numpy().squeeze()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y

def try_repo_or_load_from_file(id, name):
    try:
        fetched_data = fetch_ucirepo(id=id)
        X, y = read_uci(fetched_data)
    except ConnectionError:
        data = np.genfromtxt(f"./data/{name}.csv", delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]

    return X,y

def create_ecoli():
    X, y = try_repo_or_load_from_file(id=39, name="ecoli")

    return X, y

def create_glass():
    X, y = try_repo_or_load_from_file(id=42, name="glass")

    return X, y

def create_statlog():
    X, y = try_repo_or_load_from_file(id=147, name="statlog")

    return X, y


def create_yeast():
    X, y = try_repo_or_load_from_file(id=110, name="yeast")

    return X, y


def create_wdbc():
    X, y = try_repo_or_load_from_file(id=17, name="wdbc")

    return X, y

def create_wine():
    X, y = try_repo_or_load_from_file(id=109, name="wine")

    return X, y

def create_sonar():
    X, y = try_repo_or_load_from_file(id=151, name="sonar")

    return X, y

def create_ionosphere():
    X, y = try_repo_or_load_from_file(id=52, name="sonar")

    return X, y