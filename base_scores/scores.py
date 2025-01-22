import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score

def compute_ari(X, y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)

def compute_ami(X, y_true, y_pred):
    return adjusted_mutual_info_score(y_true, y_pred)

def compute_purity(X, y_true, y_pred):
    contingency_mat = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)

def compute_silhouette(X, y_true, y_pred):
    return silhouette_score(X, y_pred)

def compute_calinski_harabasz(X, y_true, y_pred):
    return calinski_harabasz_score(X, y_pred)

def compute_davies_bouldin(X, y_true, y_pred):
    return davies_bouldin_score(X, y_pred)

