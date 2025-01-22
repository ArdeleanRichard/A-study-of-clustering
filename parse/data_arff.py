import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder


def transform_arff_data(data):
    X = []
    y = []
    for sample in data:
        x = []
        for id, value in enumerate(sample):
            if id == len(sample) - 1:
                y.append(value)
            else:
                x.append(value)
        X.append(x)


    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return (X, y)


def create_ecoli():
    data, meta = arff.loadarff('./data/ecoli.arff')
    return transform_arff_data(data)

def create_glass():
    data, meta = arff.loadarff('./data/glass.arff')
    return transform_arff_data(data)

def create_yeast():
    data, meta = arff.loadarff('./data/yeast.arff')
    return transform_arff_data(data)