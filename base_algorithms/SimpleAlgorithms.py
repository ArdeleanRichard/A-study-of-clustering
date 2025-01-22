from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from base_algorithms.BaseClusteringAlgorithm import BaseClusteringAlgorithm

class AlgorithmKMeans(BaseClusteringAlgorithm):
    def __init__(self, n_clusters=3, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


class AlgorithmDBSCAN(BaseClusteringAlgorithm):
    def __init__(self, eps=0.5, min_samples=5):
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = None

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_  # Store the labels_

    def predict(self, X):
        # In DBSCAN, predict may not always be meaningful.
        # Return the labels from the last fit.
        if self.labels_ is None:
            raise ValueError("Fit the model before predicting.")
        return self.labels_