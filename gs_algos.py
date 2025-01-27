from clustpy.deep import ACeDeC, AEC, DCN, DDC, DEC, DeepECT, DipDECK, DipEncoder, DKM, ENRC, IDEC, VaDE, N2D
from clustpy.density import MultiDensityDBSCAN
from clustpy.hierarchical import Diana
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, Birch, OPTICS, HDBSCAN, AffinityPropagation
from clustpy.partition import DipInit, DipMeans, DipNSub, GapStatistic, ProjectedDipMeans, SpecialK, XMeans, SkinnyDip, PGMeans, LDAKmeans, GMeans
from clustpy.partition.subkmeans import SubKmeans
from clustpy.alternative import AutoNR, NrKmeans

from clustering_algos.DRLDBSCAN.main import DrlDbscanAlgorithm
from clustering_algos.autoclustering_pytorch import AutoClustering


def load_algorithms():
    algorithms = {
        # "kmeans": {
        #     "estimator": KMeans,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "init": ["k-means++", "random"],
        #         "max_iter": [300, 500, 1000],
        #     },
        # },
        # "dbscan": {
        #     "estimator": DBSCAN,
        #     "param_grid": {
        #         "eps": [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
        #         "min_samples": [1, 3, 5, 10, 15],
        #     },
        # },
        # "agglomerative": {
        #     "estimator": AgglomerativeClustering,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "linkage": ["ward", "complete", "average"],
        #     },
        # },
        # "spectral": {
        #     "estimator": SpectralClustering,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "affinity": ["nearest_neighbors", "rbf"],
        #         "n_neighbors": [5, 10,20],
        #         "random_state": [42],
        #     },
        # },
        # "meanshift": {
        #     "estimator": MeanShift,
        #     "param_grid": {
        #         "bandwidth": [None, 0.1, 0.2, 0.3],
        #         "bin_seeding": [True, False],
        #         "max_iter": [300, 500, 1000],
        #     },
        # },
        # "birch": {
        #     "estimator": Birch,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "threshold": [0.01, 0.025, 0.05, 0.1, 0.3, 0.5, 0.7],
        #         "branching_factor": [30, 40, 50, 60, 70],
        #     },
        # },
        # "optics": {
        #     "estimator": OPTICS,
        #     "param_grid": {
        #         "min_samples": [5, 10, 15],
        #         "xi": [0.01, 0.025, 0.05, 0.1, 0.2],
        #         "min_cluster_size": [0.01, 0.025, 0.05, 0.1, 0.2],
        #     },
        # },
        # "hdbscan": {
        #     "estimator": HDBSCAN,
        #     "param_grid": {
        #         "min_cluster_size": [5, 10, 15],
        #         "metric": ["euclidean", "manhattan"],
        #         "leaf_size": [25, 40, 70, 100]
        #     },
        # },
        # "affinity": {
        #     "estimator": AffinityPropagation,
        #     "param_grid": {
        #         "damping": [0.5, 0.7, 0.9],
        #         "affinity": ['precomputed', "euclidean"],
        #         "preference": [None, -50, -100, -150, -200, -220, -240],
        #     }
        # },





        "dipInit": {
            "estimator": DipInit,
            "param_grid": {
                "n_clusters": [2, 3, 4, 5],
                "dip_threshold": [0.1, 0.25, 0.5, 0.75, 1],
            },
        },
        # NEEDS Data to be 1-dimensional - error doesnt appear for significance = 1, but doesnt seem to end ether
        "dipNSub": {
            "estimator": DipNSub,
            "param_grid": {
                "outliers": [True, False],
                "consider_duplicates": [True, False],
                "threshold": [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1],
                "significance": [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1,],
            },
        },

        # "dipMeans": {
        #         "estimator": DipMeans,
        #         "param_grid": {
        #             "significance": [0.0005, 0.001, 0.005, 0.01, 0.1],
        #             "split_viewers_threshold": [0.005, 0.01, 0.05, 0.1],
        #             "pval_strategy": ["table", "function", "bootstrap"],
        #         },
        #     },
        # "gapStatistic": {
        #     "estimator": GapStatistic,
        #     "param_grid": {
        #         "n_boots": [5, 10, 15, 20, 30],
        #         "use_principal_components": [True, False],
        #         "use_log": [True, False],
        #     },
        # },

        # "gmeans": {
        #     "estimator": GMeans,
        #     "param_grid": {
        #         "significance": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        #         "n_clusters_init": [2],
        #         "n_split_trials": [1, 2, 5, 10, 20],
        #     },
        # },
        # "ldakmeans": {
        #     "estimator": LDAKmeans,
        #     "param_grid": {
        #         "n_clusters": [2],
        #         "max_iter": [300, 500, 1000],
        #         "n_init": [1,2,5,10],
        #     },
        # },
        # "pgmeans": {
        #     "estimator": PGMeans,
        #     "param_grid": {
        #         "significance": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        #         "n_projections": [None, 1],
        #         "n_samples": [None, 1],
        #     },
        # },
        "projectedDipMeans": {
            "estimator": ProjectedDipMeans,
            "param_grid": {
                "significance": [0.0005, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.1],
                "pval_strategy": ["table", "bootstrap", "function"],
                "n_split_trials": [5, 10, 12, 15, 18, 20, 25],
            },
        },

        # "skinnydip": {
        #     "estimator": SkinnyDip,
        #     "param_grid": {
        #         "significance": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        #         "pval_strategy": ["table", "bootstrap", "function"],
        #     },
        # },
        "specialK": {
            "estimator": SpecialK,
            "param_grid": {
                "significance": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
                "similarity_matrix": ["NAM", "SAM"],
                "n_neighbors": [3, 5, 10, 15, 20],
            },
        },
        # "subkmeans": {
        #     "estimator": SubKmeans,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "max_iter": [300,500,1000],
        #     },
        # },
        # "xmeans": {
        #     "estimator": XMeans,
        #     "param_grid": {
        #         "n_clusters_init": [2, 3, 4, 5],
        #     },
        # },



        # ### Density-based Clustering:
        # "mddbscan": {
        #     "estimator": MultiDensityDBSCAN,
        #     "param_grid": {
        #         "k": [3,5,10,15,20,30],
        #         "var": [1, 1.5, 2, 2.5, 3, 3.5, 4],
        #         "min_cluster_size": [5, 10, 15],
        #     },
        # },
        #
        # ### Hierarchical Clustering:
        # "diana": {
        #     "estimator": Diana,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #     },
        # },




        ### Alternative Clustering:
        # labels_pred must be 1D: shape is (788, 2)
        # "autoNR": {
        #     "estimator": AutoNR,
        #     "param_grid": {
        #         "nrkmeans_repetitions": [10, 15, 20],
        #         "outliers": [True, False],
        #     },
        # },
        # # 'int' object has no attribute 'copy'
        # "NR-Kmeans": {
        #     "estimator": NrKmeans,
        #     "param_grid": {
        #         "n_clusters": [2, 3, 4, 5],
        #         "outliers": [True, False],
        #         "max_iter": [200, 300, 400, 500],
        #         "n_init": [1, 2, 3],
        #     },
        # },





        ### DEEP CLUSTERINGS:

        "autoclustering": {
           "estimator": AutoClustering,
           "param_grid": {
               "n_clusters": [2],
               "input_dim": [1],
               "init": ["random"]
           },
        },

    #     "acedec": {
    #         "estimator": ACeDeC,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             "init": ["acedec", 'subkmeans', 'random', 'sgd'],
    #             "embedding_size": [20, 30, 40, 50, 60, 70],
    #             # "pretrain_optimizer_params": [{"lr": 1e-2}, {"lr": 1e-3}, {"lr": 1e-4}],
    #             # "clustering_optimizer_params": [{"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}],
    #             # "pretrain_epochs": [100, 150, 200],
    #             # "clustering_epochs": [100, 150, 200],
    #             # "batch_size": [32, 64, 128],
    #
    #         },
    #     },
    #     "aec": {
    #         "estimator": AEC,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "dcn": {
    #         "estimator": DCN,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "ddc": {
    #         "estimator": DDC,
    #         "param_grid": {
    #             "ratio": [0.01, 0.05, 0.1, 0.2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "dec": {
    #         "estimator": DEC,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "deepect": {
    #         "estimator": DeepECT,
    #         "param_grid": {
    #             "max_n_leaf_nodes": [1,2,3,4,5, 10, 20, 50, 100],
    #             # "random_state ": [42]
    #         },
    #     },
    #
    #     "dipdeck": {
    #         "estimator": DipDECK,
    #         "param_grid": {
    #             "n_clusters_init": [2,3,5,10],
    #             "dip_merge_threshold": [0.1, 0.3, 0.5, 0.7, 0.9],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "dipencoder": {
    #         "estimator": DipEncoder,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "dkm": {
    #         "estimator": DKM,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "enrc": {
    #         "estimator": ENRC,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "idec": {
    #         "estimator": IDEC,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "n2d": {
    #         "estimator": N2D,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "vade": {
    #         "estimator": VaDE,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             # "random_state ": [42]
    #         },
    #     },
    #     "autoclustering": {
    #         "estimator": AutoClustering,
    #         "param_grid": {
    #             "n_clusters": [2],
    #             "input_dim": [1],
    #             "init": ["random"]
    #         },
    #     },




    }

    return algorithms



# NOT A CLUSTERING ALGORITHM
# "drldbscan": {
#     "estimator": DrlDbscanAlgorithm,
#     "param_grid": {
#         "init": ["random"]
#     },
# },
