from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def read_uci(fetched_data):
    X = fetched_data.data.features.to_numpy()
    y = fetched_data.data.targets.to_numpy().squeeze()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y

def create_ecoli():
    fetched_data = fetch_ucirepo(id=39)
    return read_uci(fetched_data)

def create_glass():
    fetched_data = fetch_ucirepo(id=42)
    return read_uci(fetched_data)

def create_statlog():
    fetched_data = fetch_ucirepo(id=147)
    return read_uci(fetched_data)


def create_yeast():
    fetched_data = fetch_ucirepo(id=110)
    return read_uci(fetched_data)


def create_statlog():
    fetched_data = fetch_ucirepo(id=147)
    return read_uci(fetched_data)

def create_wdbc():
    fetched_data = fetch_ucirepo(id=17)
    return read_uci(fetched_data)


def create_wine():
    fetched_data = fetch_ucirepo(id=109)
    return read_uci(fetched_data)


def create_sonar():
    fetched_data = fetch_ucirepo(id=151)
    return read_uci(fetched_data)

def create_ionosphere():
    fetched_data = fetch_ucirepo(id=52)
    return read_uci(fetched_data)