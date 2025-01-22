import numpy as np
from sklearn import datasets
from base_datasets.BaseDataset import BaseDataset

class DatasetD1(BaseDataset):
    def __init__(self, n_samples=1000, seed=30, random_state=170):
        self.n_samples = n_samples
        self.seed = seed
        self.random_state = random_state

    def load_data(self):
        return datasets.make_blobs(n_samples=self.n_samples, random_state=self.seed)

class DatasetD2(BaseDataset):
    def __init__(self, n_samples=1000, seed=30, random_state=170):
        self.n_samples = n_samples
        self.seed = seed
        self.random_state = random_state

    def load_data(self):
        X, y = datasets.make_blobs(n_samples=self.n_samples, cluster_std=1.0, random_state=self.random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        data = (X_aniso, y)
        return data


class DatasetD3(BaseDataset):
    def __init__(self, n_samples=1000, seed=30, random_state=170, n_features=2):
        self.n_samples = n_samples
        self.seed = seed
        self.random_state = random_state
        self.n_features = n_features

    def load_data(self):
        return datasets.make_blobs(n_samples=self.n_samples, n_features=self.n_features, cluster_std=[1.0, 2.5, 0.5], random_state=self.random_state)

class DatasetD4(BaseDataset):
    def __init__(self, n_samples=1000, seed=30, random_state=170):
        self.n_samples = n_samples
        self.seed = seed
        self.random_state = random_state

    def load_data(self):
        return datasets.make_moons(n_samples=self.n_samples, noise=0.05, random_state=self.seed)

class DatasetD5(BaseDataset):
    def __init__(self, n_samples=1000, seed=30, random_state=170):
        self.n_samples = n_samples
        self.seed = seed
        self.random_state = random_state

    def load_data(self):
        return datasets.make_circles(n_samples=self.n_samples, factor=0.5, noise=0.05, random_state=self.seed)
