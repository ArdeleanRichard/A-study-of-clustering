import itertools
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix

from constants import DIR_RESULTS
from gs_algos import load_algorithms
from gs_datasets import load_all_data, load_sklearn_data_3_multiple_dimensions



# RUNS REQUIRED:
# RICI: [RUNNING] PYCLUSTERING only rock algorithm from unbalance-> as it seems to have blocked

# CHECK ALL CSV FOR ALGORITHMS WITH LOW PERF ON D1 - might need to rerun
# - ENRC seems to not run on anything
# - Might require parameter reruns:
# -- affinity
# -- clique
# -- cure
# -- dipInit
# -- skinnydip
# -- syncsom
# -- ttsas

def normalize_dbs(df):
    # df['davies_bouldin_score'] = 1 - df['davies_bouldin_score'] / df['davies_bouldin_score'].max() # doesnt work as well as I would like
    df['norm_davies_bouldin_score'] = 1 / (1 + df['davies_bouldin_score'])
    return df

# def normalize_chs(df):
#     df['calinski_harabasz_score'] = np.log1p(df['calinski_harabasz_score'])
#     df['calinski_harabasz_score'] = df['calinski_harabasz_score'] / df['calinski_harabasz_score'].max()
#     return df

# def normalize_dbs_chs(df):
#     df = normalize_chs(df)
#     df = normalize_dbs(df)
#     return df

def perform_grid_search(datasets, algorithms, n_repeats=10, multiple_dimensions=False):
    for dataset_name, (X, y_true) in datasets:
        # scale all datasets to the same range
        scaler = preprocessing.MinMaxScaler().fit(X)
        X = scaler.transform(X)

        for algo_name, algo_details in algorithms.items():
            results = []
            param_names = list(algo_details["param_grid"].keys())

            # -------------
            # SPECIAL PARAMS
            # -------------
            for param_name in param_names:
                if param_name == "n_clusters" or param_name == "number_clusters" or param_name == "n_clusters_init" or param_name == "amount_clusters":
                    algo_details["param_grid"][param_name] = [len(np.unique(y_true))]
                if param_name == "input_dim":
                    algo_details["param_grid"][param_name] = [X.shape[1]]
                if param_name == "maximum_clusters":
                    algo_details["param_grid"][param_name] = [len(np.unique(y_true))+1]
                if param_name == "bandwidth":
                    bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=50)
                    algo_details["param_grid"][param_name].extend([bandwidth])
            # -------------


            param_combinations = list(itertools.product(*algo_details["param_grid"].values()))
            print(dataset_name, algo_name)

            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                # is_nondeterministic = any(
                #     key in param_dict for key in ["init", "random_state"]
                # )
                is_nondeterministic = False

                scores_per_repeat = []

                for _ in range(n_repeats if is_nondeterministic else 1):
                    try:
                        estimator = algo_details["estimator"](**param_dict)
                        y_pred = estimator.fit_predict(X)

                        if len(np.unique(y_pred)) > 1:  # Ensure more than one cluster
                            ari = adjusted_rand_score(y_true, y_pred)
                            ami = adjusted_mutual_info_score(y_true, y_pred)
                            contingency_mat = contingency_matrix(y_true, y_pred)
                            purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
                            silhouette = silhouette_score(X, y_pred)
                            calinski_harabasz = calinski_harabasz_score(X, y_pred)
                            davies_bouldin = davies_bouldin_score(X, y_pred)
                        else:
                            print(f"[1CLUST] {algo_name}, {params}")
                            ari = ami = purity = silhouette = calinski_harabasz = davies_bouldin = -1

                        scores_per_repeat.append({
                            "adjusted_rand_score": ari,
                            "adjusted_mutual_info_score": ami,
                            "purity_score": purity,
                            "silhouette_score": silhouette,
                            "calinski_harabasz_score": calinski_harabasz,
                            "davies_bouldin_score": davies_bouldin,
                        })
                        print(param_dict)

                    except Exception as e:
                        print(f"[ERROR] {algo_name}, {params}, {e}")
                        scores_per_repeat.append({
                            "adjusted_rand_score": -1,
                            "adjusted_mutual_info_score": -1,
                            "purity_score": -1,
                            "silhouette_score": -1,
                            "calinski_harabasz_score": -1,
                            "davies_bouldin_score": -1,
                        })

                # Aggregate scores across repeats for nondeterministic algorithms
                if is_nondeterministic:
                    aggregated_scores = {
                        key: np.nanmean([score[key] for score in scores_per_repeat])
                        for key in scores_per_repeat[0]
                    }
                else:
                    aggregated_scores = scores_per_repeat[0]

                results.append({
                    **param_dict,
                    **aggregated_scores,
                })

            # Save results to CSV
            results_df = pd.DataFrame(results)
            results_df = normalize_dbs(results_df)
            path = DIR_RESULTS + f"/grid_search/{algo_name}_{dataset_name}.csv" if not multiple_dimensions \
                else DIR_RESULTS + f"/grid_search/multiple_dimensions/{algo_name}_{dataset_name}.csv"
            results_df.to_csv(path, index=False)


if __name__ == "__main__":
    datasets = load_all_data()
    # datasets = load_sklearn_data_3_multiple_dimensions()
    algorithms = load_algorithms()
    perform_grid_search(datasets, algorithms)
