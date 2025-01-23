import json
from importlib import import_module

def load_class(module_name, class_name):
    """
    Dynamically load a class from a module.
    :param module_name: Module name (e.g., 'datasets.synthetic').
    :param class_name: Class name (e.g., 'SyntheticDataset').
    :return: Class object.
    """
    module = import_module(module_name)
    return getattr(module, class_name)

def run_experiment(experiment):
    """
    Run a single experiment based on its configuration.
    """
    # Load dataset
    dataset_cfg = experiment["dataset"]
    dataset_name = dataset_cfg["name"]
    dataset_params = dataset_cfg["params"]

    dataset_class = load_class("base_datasets", dataset_name)
    dataset = dataset_class(**dataset_params)
    X, y = dataset.load_data()

    # Load algorithm
    algorithm_cfg = experiment["algorithm"]
    algorithm_class = load_class("base_algorithms", algorithm_cfg["name"])
    algorithm = algorithm_class(**algorithm_cfg["params"])
    algorithm.fit(X)
    labels = algorithm.predict(X)

    # Compute metric
    metric_name = experiment["score"]
    metric_function = load_class("base_scores", metric_name)
    score = metric_function(X, y, labels)

    return {
        "dataset": dataset_cfg,
        "algorithm": algorithm_cfg,
        "score_name": metric_name,
        "score_value": score
    }
def main(config_file="./configs/experiment_config.json"):
    """
    Main function to execute all experiments from a configuration file.
    :param config_file: Path to the JSON configuration file.
    """
    with open(config_file, "r") as f:
        config = json.load(f)

    experiments = config.get("experiments", [])
    results = []

    for experiment in experiments:
        print(f"Running experiment: {experiment}")
        try:
            result = run_experiment(experiment)
            results.append(result)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Experiment failed: {experiment}")
            print(f"Error: {e}")

        print()

    # Save results to a file
    with open("../results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("All experiments completed. Results saved to 'results.json'.")

if __name__ == "__main__":
    main()