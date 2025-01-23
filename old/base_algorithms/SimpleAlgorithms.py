from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, Birch, OPTICS, HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from old.base_algorithms import BaseClusteringAlgorithm


class AlgorithmKMeans(BaseClusteringAlgorithm):
    def __init__(self, n_clusters=3, random_state=42, init='k-means++', max_iter=300):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, init=init, max_iter=max_iter)

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


class AlgorithmAgglomerativeClustering(BaseClusteringAlgorithm):
    def __init__(self, n_clusters=3, linkage='ward'):
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.labels_


class AlgorithmSpectralClustering(BaseClusteringAlgorithm):
    def __init__(self, n_clusters=3, affinity='nearest_neighbors', random_state=42):
        self.model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=random_state)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.labels_


class AlgorithmMeanShift(BaseClusteringAlgorithm):
    def __init__(self, bandwidth=None, bin_seeding=False):
        self.model = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


class AlgorithmBirch(BaseClusteringAlgorithm):
    def __init__(self, n_clusters=3, threshold=0.5, branching_factor=50):
        self.model = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


class AlgorithmOPTICS(BaseClusteringAlgorithm):
    def __init__(self, min_samples=5, xi=0.05, min_cluster_size=0.05):
        self.model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        self.labels_ = None

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_

    def predict(self, X):
        if self.labels_ is None:
            raise ValueError("Fit the model before predicting.")
        return self.labels_




class AlgorithmHDBSCAN(BaseClusteringAlgorithm):
    def __init__(self,min_cluster_size=5, leaf_size=40, metric="euclidean"):
        self.model = HDBSCAN(min_cluster_size=min_cluster_size, leaf_size=leaf_size, metric=metric)
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



class AlgorithmAffinityPropagation(BaseClusteringAlgorithm):
    def __init__(self, damping=0.9, preference=None):
        self.model = AffinityPropagation(damping=damping, preference=preference)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


