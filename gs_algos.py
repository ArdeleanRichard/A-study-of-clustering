from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, Birch, OPTICS, HDBSCAN, AffinityPropagation


def load_algorithms():
    # Step 2: Define algorithms and their parameter grids
    algorithms = {
        "kmeans": {
            "estimator": KMeans,
            "param_grid": {
                "n_clusters": [2, 3, 4, 5],
                "init": ["k-means++", "random"],
                "max_iter": [300, 500],
            },
        },
        # "dbscan": {
        #     "estimator": DBSCAN,
        #     "param_grid": {
        #         "eps": [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
        #         "min_samples": [1, 3, 5, 10, 15],
        #     },
        # },
    #     "agglomerative": {
    #         "estimator": AgglomerativeClustering,
    #         "param_grid": {
    #             "n_clusters": [2, 3, 4, 5],
    #             "linkage": ["ward", "complete", "average"],
    #         },
    #     },
    #     "spectral": {
    #         "estimator": SpectralClustering,
    #         "param_grid": {
    #             "n_clusters": [2, 3, 4, 5],
    #             "affinity": ["nearest_neighbors", "rbf"],
    #             "random_state": [42],
    #         },
    #     },
    #     "meanshift": {
    #         "estimator": MeanShift,
    #         "param_grid": {
    #             "bandwidth": [None, 0.1, 0.2, 0.3],
    #             "bin_seeding": [True, False],
    #         },
    #     },
    #     "birch": {
    #         "estimator": Birch,
    #         "param_grid": {
    #             "n_clusters": [2, 3, 4, 5],
    #             "threshold": [0.3, 0.5, 0.7],
    #             "branching_factor": [30, 50, 70],
    #         },
    #     },
    #     "optics": {
    #         "estimator": OPTICS,
    #         "param_grid": {
    #             "min_samples": [5, 10, 15],
    #             "xi": [0.05, 0.1],
    #             "min_cluster_size": [0.05, 0.1],
    #         },
    #     },
    #     "hdbscan": {
    #         "estimator": HDBSCAN,
    #         "param_grid": {
    #             "min_cluster_size": [5, 10, 15],
    #             "metric": ["euclidean", "manhattan"],
    #             "leaf_size": [25, 40, 70, 100]
    #         },
    #     },
    #     "affinity": {
    #         "estimator": AffinityPropagation,
    #         "param_grid": {
    #             "damping": [0.5, 0.7, 0.9],
    #             "preference": [None, -50, -100],
    #         },
    #     },

    }
    return algorithms


