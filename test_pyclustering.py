from matplotlib import pyplot as plt
from pyclustering.cluster.bang import bang
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
import numpy as np
import warnings

from clustering_algos.Algorithms import create_algorithm_wrapper

np.warnings = warnings
import visualization.scatter_plot as sp

def converter_clusters_to_labels(X, clusters):
    alen = len(X)
    labels = np.zeros((alen, ))
    current_label = 0
    for cluster in clusters:
        for cluster_point in cluster:
            labels[cluster_point] = current_label
        current_label+=1

    return labels

# Load list of points for cluster analysis.
sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)
#
#
# # Prepare initial centers using K-Means++ method.
# initial_centers = kmeans_plusplus_initializer(sample, 2).initialize()
#
# # Create instance of K-Means algorithm with prepared centers.
# kmeans_instance = kmeans(sample, initial_centers)
#
# # Run cluster analysis and obtain results.
# kmeans_instance.process()
# clusters = kmeans_instance.get_clusters()
# print(clusters)
# final_centers = kmeans_instance.get_centers()
#
# labels = converter_clusters_to_labels(sample, clusters)
#
#
# sample = np.array(sample)
# sp.plot("test", sample, labels)
# plt.show()


# Create instance of K-Means algorithm with prepared centers.
kmeans_instance = bang(sample, levels=12)

# Run cluster analysis and obtain results.
kmeans_instance.process()
clusters = kmeans_instance.get_clusters()
print(clusters)

labels = converter_clusters_to_labels(sample, clusters)

sample = np.array(sample)
sp.plot("test", sample, labels)
plt.show()


BANG = create_algorithm_wrapper(bang)
model = BANG(levels=15)

labels = model.fit_predict(sample)

sample = np.array(sample)
sp.plot("test", sample, labels)
plt.show()
