import ast
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix

from constants import DIR_RESULTS


from gs_algos import load_algorithms
from gs_datasets import load_sklearn_data_3_multiple_dimensions, load_sklearn_data_3_multiple_samples


def perform_analysis_with_best_params(datasets, algorithms, n_repeats=10, result_file="dimensional_analysis_results"):
    """
    Perform analysis using the best parameters for each algorithm on datasets with
    varying dimensionality and sample sizes, and append results to an existing CSV.
    Results are saved incrementally after each algorithm run.

    Parameters:
    -----------
    datasets : list of tuples
        Each tuple contains (dataset_name, (X, y_true)) where X is the feature matrix
        and y_true are the ground truth labels.
    algorithms : dict
        Dictionary of algorithms with their estimators.
    n_repeats : int, default=10
        Number of repetitions for non-deterministic algorithms.
    result_file : str, default="dimensional_analysis_results"
        Name of the result file without extension.
    """
    # Load best parameters for D3 dataset from the CSV file
    best_params_df = pd.read_csv("./paper/best_params.csv")
    d3_params = best_params_df[best_params_df['dataset'] == 'D3']

    # Convert to dictionary keyed by algorithm
    best_params = {}
    for _, row in d3_params.iterrows():
        algo = row['algorithm']
        # Extract all non-null parameters for this algorithm
        param_dict = {col: row[col] for col in d3_params.columns
                      if col not in ['algorithm', 'dataset'] and not pd.isna(row[col])}
        best_params[algo] = param_dict

    # Define CSV file path
    csv_path = f"{DIR_RESULTS}/{result_file}.csv"

    # Check if the file exists, if not create it with headers
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        columns = ["dataset", "algorithm", "dimensions", "n_samples", "execution_time",
                   "adjusted_rand_score", "adjusted_mutual_info_score", "purity_score",
                   "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
        print(f"Created new results file: {csv_path}")

    for dataset_name, (X, y_true) in datasets:
        # Scale dataset
        scaler = preprocessing.MinMaxScaler().fit(X)
        X = scaler.transform(X)

        dimensions = X.shape[1]  # Get dimensionality
        n_samples = X.shape[0]  # Get number of samples

        print(f"Processing dataset: {dataset_name}, Dimensions: {dimensions}, Samples: {n_samples}")

        for algo_name, algo_details in algorithms.items():
            # For algorithms not in D3 best params, skip or use default
            if algo_name not in best_params:
                print(f"No best parameters found for {algo_name} on D3, skipping...")
                continue

            # Get best parameters for this algorithm
            param_dict = best_params[algo_name].copy()

            # Handle special parameters that depend on the dataset
            for param_name in list(param_dict.keys()):
                if param_name in ["n_clusters", "number_cluster", "number_clusters",
                                  "n_clusters_init", "amount_clusters"]:
                    param_dict[param_name] = len(np.unique(y_true))
                elif param_name == "input_dim":
                    param_dict[param_name] = X.shape[1]
                elif param_name in ["min_n_clusters"]:
                    param_dict[param_name] = len(np.unique(y_true)) - 1
                elif param_name in ["maximum_clusters", "max_n_clusters"]:
                    param_dict[param_name] = len(np.unique(y_true)) + 1
                elif param_name == "bandwidth":
                    param_dict[param_name] = estimate_bandwidth(X, quantile=0.1, n_samples=50)

            # Get the proper parameter types for this algorithm
            param_dict = convert_param_types(param_dict)

            print(f"Running {algo_name} with parameters: {param_dict}")

            # Check if algorithm is non-deterministic
            is_nondeterministic = True

            execution_times = []
            scores_per_repeat = []

            for _ in range(n_repeats if is_nondeterministic else 1):
                try:
                    # Measure execution time
                    start_time = time.time()

                    estimator = algo_details["estimator"](**param_dict)
                    y_pred = estimator.fit_predict(X)

                    end_time = time.time()
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)

                    # Calculate metrics
                    if len(np.unique(y_pred)) > 1:  # Ensure more than one cluster
                        ari = adjusted_rand_score(y_true, y_pred)
                        ami = adjusted_mutual_info_score(y_true, y_pred)
                        contingency_mat = contingency_matrix(y_true, y_pred)
                        purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
                        silhouette = silhouette_score(X, y_pred)
                        calinski_harabasz = calinski_harabasz_score(X, y_pred)
                        davies_bouldin = davies_bouldin_score(X, y_pred)
                    else:
                        print(f"[1CLUST] {algo_name}")
                        ari = ami = purity = silhouette = calinski_harabasz = davies_bouldin = -1

                    scores_per_repeat.append({
                        "adjusted_rand_score": ari,
                        "adjusted_mutual_info_score": ami,
                        "purity_score": purity,
                        "silhouette_score": silhouette,
                        "calinski_harabasz_score": calinski_harabasz,
                        "davies_bouldin_score": davies_bouldin,
                    })

                except Exception as e:
                    print(f"[ERROR] {algo_name}, {e}")
                    execution_times.append(-1)
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
                mean_execution_time = np.nanmean(execution_times)
            else:
                aggregated_scores = scores_per_repeat[0]
                mean_execution_time = execution_times[0]

            # Create a single result for this dataset-algorithm pair
            result = {
                "dataset": dataset_name,
                "algorithm": algo_name,
                "dimensions": dimensions,
                "n_samples": n_samples,
                "execution_time": mean_execution_time,
                **aggregated_scores
            }

            # Create a DataFrame for this single result
            result_df = pd.DataFrame([result])

            try:
                # Append this single result to the CSV file
                result_df.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"Result for {dataset_name} - {algo_name} appended to: {csv_path}")
            except Exception as e:
                print(f"Error when saving result for {dataset_name} - {algo_name}: {e}")
                # Fallback - save to a different filename
                fallback_path = f"{DIR_RESULTS}/{result_file}_fallback.csv"
                result_df.to_csv(fallback_path, mode='a', header=not os.path.exists(fallback_path), index=False)
                print(f"Result saved to fallback file: {fallback_path}")

    print(f"Analysis completed with results saved incrementally to: {csv_path}")

    # Read the full results for return value
    try:
        return pd.read_csv(csv_path)
    except:
        print(f"Warning: Could not read the final results file")
        return None


def convert_or_return_string(value):
    try:
        # Try to safely evaluate the string as a Python literal
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
        # Try converting the evaluated value to a dict (like from list of tuples)
        return dict(parsed)
    except (ValueError, SyntaxError, TypeError):
        # Return original string if it can't be converted
        return value

def convert_param_types(param_dict):
    """
    Convert parameter values to appropriate types based on estimator requirements.

    Parameters:
    -----------
    param_dict : dict
        Dictionary of parameters.
    estimator_class : class
        Estimator class.

    Returns:
    --------
    dict
        Updated parameter dictionary with correct types.
    """
    converted_params = {}

    # Known parameter types that need special handling
    int_params = ["max_iter", "random_state", "n_init", "pretrain_epochs",
                  "clustering_epochs", "min_samples", "input_dim", "embedding_size",
                  "n_neighbors", "branching_factor", "leaf_size", "convergence_iter",
                  "min_cluster_size", "n_boots"]
    float_params = ["eps", "bandwidth", "damping"]
    bool_params = ["bin_seeding", "add_tails"]

    for key, value in param_dict.items():
        test = convert_or_return_string(value)
        if isinstance(test, dict):
            converted_params[key] = test
            continue
        if isinstance(value, (int, float, bool, str)):
            if isinstance(value, float):
                if value.is_integer():
                    converted_params[key] = int(value)
                else:
                    converted_params[key] = value
            elif key in bool_params:
                if isinstance(value, str):
                    converted_params[key] = value.lower() in ['true', 'yes', '1']
                else:
                    converted_params[key] = bool(value)
            else:
                # Keep as is
                converted_params[key] = value
        else:
            # Non-primitive types, keep as is
            converted_params[key] = value

    return converted_params


def multi_dimensional_analysis():
    """
    Perform analysis on datasets with varying dimensionality and sample sizes
    using the best parameters for each algorithm on D3.
    """
    datasets = load_sklearn_data_3_multiple_dimensions()
    algorithms = load_algorithms()
    results = perform_analysis_with_best_params(datasets, algorithms, n_repeats=5, result_file="time_analysis_dimensions")

    return results


def multi_samples_analysis():
    """
    Perform analysis on datasets with varying dimensionality and sample sizes
    using the best parameters for each algorithm on D3.
    """
    datasets = load_sklearn_data_3_multiple_samples()
    algorithms = load_algorithms()
    results = perform_analysis_with_best_params(datasets, algorithms, n_repeats=10, result_file="time_analysis_samples")

    return results


if __name__ == "__main__":
    multi_dimensional_analysis()
    # multi_samples_analysis()