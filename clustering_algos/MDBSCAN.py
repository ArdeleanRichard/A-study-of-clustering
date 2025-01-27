def snnc_clustering(data_points, k_nearest_neighbors):
    """
    Share Nearest Neighbors-based Clustering method (SNNC).

    Parameters:
        data_points: list of low-density data points.
        k_nearest_neighbors: function or dictionary that returns the k-nearest neighbors for a given data point.

    Returns:
        F: list of low-density natural clusters.
    """
    # Initialize variables
    num = len(data_points)  # The number of low-density points
    clusters = {i: {i} for i in range(num)}  # Initial clusters, one for each data point

    # Step 1: Merge based on shared nearest neighbors
    for i in range(num):
        for j in range(i + 1, num):
            if len(k_nearest_neighbors[i] & k_nearest_neighbors[j]) >= 1:  # Intersection of neighbors
                clusters[i].update(clusters[j])
                clusters[j] = set()  # Empty cluster j

    # Step 2: Iteratively merge clusters
    for t in range(100):
        merged = False
        for i in range(num):
            for j in range(i + 1, num):
                if len(clusters[i] & clusters[j]) >= 1:  # Shared members between clusters
                    clusters[i].update(clusters[j])
                    clusters[j] = set()  # Empty cluster j
                    merged = True
        if not merged:
            break  # Exit loop if no clusters were merged

    # Step 3: Calculate mean cluster size for non-empty clusters
    cluster_sizes = [len(c) for c in clusters.values() if len(c) > 0]
    mean_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0

    # Step 4: Add large clusters to final result F
    F = []
    for c in clusters.values():
        if len(c) >= mean_size:
            F.append(c)

    return F


# Example usage:
# Assume `data_points` is a list of data points and `knn_func` is a function that returns the k-nearest neighbors for a given point.
# Example of k_nearest_neighbors: {0: {1, 2}, 1: {0, 3}, 2: {0, 3}, 3: {1, 2}}
data_points = [...]  # List of low-density data points
k_nearest_neighbors = {i: set() for i in range(len(data_points))}  # Replace with actual k-NN sets

clusters = snnc_clustering(data_points, k_nearest_neighbors)
print(clusters)