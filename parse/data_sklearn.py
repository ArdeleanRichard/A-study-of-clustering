import numpy as np
from sklearn import datasets

seed = 30
random_state = 170

def create_data1(n_samples):
    return datasets.make_blobs(n_samples=n_samples, random_state=seed)

def create_data2(n_samples):
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, cluster_std=1.0, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    return aniso

def create_data3(n_samples, n_features=2):
    # data5 with data3 variances
    return datasets.make_blobs(n_samples=n_samples, n_features=n_features, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)


def create_data4(n_samples):
    return datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)


def create_data5(n_samples):
    return datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)