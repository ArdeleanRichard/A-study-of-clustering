import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, KMeans

from data_read.data_arff import create_compound
from data_read.data_sklearn import create_data1
import visualization.scatter_plot as sp



class AMDDBSCAN:
    def __init__(self):
        pass

    def obtain_eps_list(self, data):
        """
        Compute the Eps list from the dataset.

        Parameters:
            data (ndarray): A 2D numpy array where each row represents a point in the dataset.

        Returns:
            list: The Eps list computed from the dataset.
        """
        # Compute the pairwise Euclidean distance matrix
        DIST = cdist(data, data, metric='euclidean')

        # Sort each row in ascending order
        SORTED_DIST = np.sort(DIST, axis=1)

        # Compute the mean of each column in the sorted matrix to obtain eps_list
        eps_list = np.mean(SORTED_DIST[:, 1:], axis=0) # indexing removes 0 from first index

        return eps_list


    def obtain_min_pts_list(self, data, eps_list):
        """
        Compute the min_pts list from the dataset and EpsList.

        Parameters:
            data (ndarray): A 2D numpy array where each row represents a point in the dataset.
            eps_list (list): A list of Eps values.

        Returns:
            list: The min_pts list computed from the dataset and EpsList.
        """
        DIST = cdist(data, data, metric='euclidean')
        min_pts_list = []

        for eps in eps_list:
            neighbors_count = np.sum(DIST <= eps, axis=1) - 1  # Exclude self-count
            min_pts = np.mean(neighbors_count)
            min_pts_list.append(round(min_pts))

        return np.array(min_pts_list)

    def run_dbscan(self, data, eps, min_pts):
        clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(data)
        return len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)


    def parameter_adaptation(self, data):
        eps_list = self.obtain_eps_list(data)
        min_pts_list = self.obtain_min_pts_list(data, eps_list)
        counter = 0

        nr_clusters_initial = self.run_dbscan(data, eps_list[1], min_pts_list[1])

        COUNTER_THRESHOLD = 3
        for i in range(len(data) - 1):
            nr_clusters_current = self.run_dbscan(data, eps_list[i], min_pts_list[i])

            if nr_clusters_current == nr_clusters_initial: # MISTAKE IN PAPER, YOU CANNOT DO IT WITH I+1, IT IS NEVER ASSIGNED
                counter += 1

                if counter > COUNTER_THRESHOLD:
                    n = nr_clusters_current
                    left, right = i, len(data) - 2
                    best_index = left

                    while left <= right:  # MISTAKE IN PAPER, START AND END DO NOT EXIST IN PAPER
                        mid = (left + right) // 2
                        nr_clusters_best = self.run_dbscan(data, eps_list[mid], min_pts_list[mid])

                        if nr_clusters_best < n:
                            right = mid
                        elif nr_clusters_best > n:
                            left = mid
                        else:
                            best_index = mid
                            break

                    return best_index

            else:
                nr_clusters_initial = nr_clusters_current
                counter = 0



        return None  # If no suitable k is found


    def obtain_candidate_eps_list(self, data):
        """
        Compute the candidate Eps list using kdis frequency histogram and K-means clustering.

        Parameters:
            data (ndarray): A 2D numpy array where each row represents a point in the dataset.

        Returns:
            list: The candidate Eps list.
        """
        # Compute the pairwise Euclidean distance matrix
        DIST = cdist(data, data, metric='euclidean')

        # Sort each row in ascending order
        SORTED_DIST = np.sort(DIST, axis=1)

        # Determine k using adaptive method (assumed to be len(data) // 10 for now)
        k = self.parameter_adaptation(data)
        kdis = SORTED_DIST[:, k]  # Extract the k-distance values

        # Create a histogram of kdis values
        hist, bin_edges = np.histogram(kdis, bins='auto')
        N = len([b for b in hist if b > np.mean(hist)])  # Estimate peaks as bins with high frequency

        # # Plot histogram
        # plt.figure(figsize=(8, 5))
        # plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
        # plt.xlabel("k-distance values")
        # plt.ylabel("Frequency")
        # plt.title("k-distance Frequency Histogram")
        # plt.show()

        # Apply K-means clustering to find candidate Eps
        kdis_reshaped = kdis.reshape(-1, 1)
        kmeans = KMeans(n_clusters=N, random_state=42).fit(kdis_reshaped)
        candidate_eps_list = kmeans.cluster_centers_.flatten().tolist()

        return candidate_eps_list


    def fit_predict(self, data):
        """
        Perform Multi-density DBSCAN clustering using candidate Eps values.

        Parameters:
            data (ndarray): A 2D numpy array where each row represents a point in the dataset.

        Returns:
            list: Cluster labels for each data point.
        """
        eps_list = self.obtain_candidate_eps_list(data)
        eps_list.sort()
        # print(eps_list)
        labels = np.full(len(data), -1)
        cluster_id = 0



        for eps in eps_list:
            min_pts_list = self.obtain_min_pts_list(data, [eps])
            dbscan = DBSCAN(eps=eps, min_samples=int(min_pts_list[0])).fit(data)
            new_labels = dbscan.labels_

            # sp.plot(f"eps {eps}", data, new_labels)

            for i in range(len(data)):
                if labels[i] == -1 and new_labels[i] != -1:
                    labels[i] = cluster_id + new_labels[i] + 1
            cluster_id = max(labels) + 1 if max(labels) != -1 else cluster_id
            data = data[new_labels == -1]

        return labels



# Example usage
if __name__ == "__main__":
    import os
    os.chdir("../")
    n_samples = 1000
    X, y = create_compound()
    scaler = preprocessing.MinMaxScaler().fit(X)
    X = scaler.transform(X)

    model = AMDDBSCAN()
    labels = model.fit_predict(X)
    sp.plot("ground truth", X, y)
    sp.plot("mdbscan", X, labels)
    plt.show()
