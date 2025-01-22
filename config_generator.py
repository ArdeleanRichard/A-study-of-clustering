import json
import itertools

def generate_experiment_config(datasets, algorithms_by_dataset, scores, output_file="experiment_config.json"):
    """
    Generates a configuration file for clustering experiments, handling multiple datasets and dataset-specific algorithm parameters.

    :param datasets: List of dataset configurations.
    :param algorithms_by_dataset: Dictionary mapping dataset names to their algorithm configurations.
    :param scores: List of metrics to compute.
    :param output_file: Path to save the JSON configuration.
    """
    experiments = []

    for dataset in datasets:
        dataset_name = dataset["name"]
        dataset_params = dataset["params"]

        # Get the algorithms for this dataset
        dataset_algorithms = algorithms_by_dataset.get(dataset_name, [])
        for algorithm, score in itertools.product(dataset_algorithms, scores):
            experiments.append({
                "dataset": {
                    "name": dataset_name,
                    "params": dataset_params
                },
                "algorithm": algorithm,
                "score": score
            })

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump({"experiments": experiments}, f, indent=4)

    print(f"Experiment configuration saved to {output_file}")

if __name__ == "__main__":
    # Example datasets
    datasets = [
        {"name": "DatasetD1", "params": {"n_samples": 1000, }},
        {"name": "DatasetD2", "params": {"n_samples": 1000, }},
    ]

    # Algorithms tailored for specific datasets
    algorithms_by_dataset = {
        "DatasetD1": [
            {"name": "AlgorithmKMeans", "params": {"n_clusters": 3}},
            {"name": "AlgorithmDBSCAN", "params": {"eps": 0.3, "min_samples": 10}}
        ],
        "DatasetD2": [
            {"name": "AlgorithmKMeans", "params": {"n_clusters": 3}},
            {"name": "AlgorithmDBSCAN", "params": {"eps": 0.3, "min_samples": 10}}
        ],
    }

    # scores to compute
    scores = ["compute_silhouette", "compute_calinski_harabasz", "compute_davies_bouldin"]

    # Generate configuration
    generate_experiment_config(datasets, algorithms_by_dataset, scores)
