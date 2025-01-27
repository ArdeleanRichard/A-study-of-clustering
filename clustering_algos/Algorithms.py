from matplotlib import pyplot as plt
from pyclustering.cluster.bang import bang
import numpy as np
import warnings
np.warnings = warnings

def converter_clusters_to_labels(X, clusters):
    alen = len(X)
    labels = np.zeros((alen, ))
    current_label = 0
    for cluster in clusters:
        for cluster_point in cluster:
            labels[cluster_point] = current_label
        current_label+=1

    return labels

# class AlgorithmBANG:
#     def __init__(self, levels):
#         self.levels = levels
#     def fit_predict(self, X):
#         # Create instance of K-Means algorithm with prepared centers.
#         alg_instance = bang(X, levels=12)
#
#         # Run cluster analysis and obtain results.
#         alg_instance.process()
#         clusters = alg_instance.get_clusters()
#         labels = converter_clusters_to_labels(X, clusters)
#         return labels

def create_algorithm_wrapper(algorithm):
    """
    Factory function to create a wrapper class for clustering algorithms.

    Parameters:
    - algorithm: A callable representing the clustering algorithm.

    Returns:
    - A dynamically created wrapper class.
    """

    class AlgorithmWrapper:
        def __init__(self, **kwargs):
            self.algorithm_args = {**kwargs}

        def fit_predict(self, X):
            # Instantiate and run the algorithm
            alg_instance = algorithm(X, **self.algorithm_args)
            alg_instance.process()
            clusters = alg_instance.get_clusters()

            labels = converter_clusters_to_labels(X, clusters)
            return labels

    return AlgorithmWrapper