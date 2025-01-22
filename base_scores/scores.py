import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def compute_silhouette(X, labels):
    return silhouette_score(X, labels)

def compute_calinski_harabasz(X, labels):
    return calinski_harabasz_score(X, labels)

def compute_davies_bouldin(X, labels):
    return davies_bouldin_score(X, labels)

def compute_purity(y_true, y_pred):
    contingency_mat = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
