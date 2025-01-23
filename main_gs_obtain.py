import os
import pandas as pd

from constants import DIR_RESULTS


# Function to compute priority score
def compute_priority_score(row):
    # Example weighted combination of metrics
    return (
        0.166 * row['silhouette_score'] +
        0.166 * row['calinski_harabasz_score'] -
        0.166 * row['davies_bouldin_score'] +
        0.166 * row['adjusted_rand_score'] +
        0.166 * row['adjusted_mutual_info_score'] +
        0.166 * row['purity_score']
    )

# Function to extract the best parameters from each file
def extract_best_parameters(results_dir):
    summary = []

    # Iterate through all CSV files in the results directory
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv"):
            algo_dataset = filename.replace(".csv", "")  # Extract "algorithm_dataset"
            print(algo_dataset.split("_"))
            algorithm, dataset = algo_dataset.split("_")

            # Load the results
            file_path = os.path.join(results_dir, filename)
            df = pd.read_csv(file_path)

            # Ensure all necessary scores are available
            required_scores = [
                "adjusted_rand_score",
                "adjusted_mutual_info_score",
                "purity_score",
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score",
            ]
            if not all(score in df.columns for score in required_scores):
                print(f"Skipping {filename}: Missing required scores.")
                continue

            # Drop rows with NaN in the required scores
            df = df.dropna(subset=required_scores)

            # Compute priority score for each row
            df['priority_score'] = df.apply(compute_priority_score, axis=1)

            # Find the row with the best priority score
            best_row = df.loc[df['priority_score'].idxmax()]

            # Save the best parameters and corresponding scores
            best_params = {
                "algorithm": algorithm,
                "dataset": dataset,
                **best_row.to_dict(),  # Include all parameters and scores
            }
            summary.append(best_params)

    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary)

    # Save the summary to a new CSV file
    summary_df.to_csv(DIR_RESULTS+"grid_search_summary.csv", index=False)

    return summary_df

# Run the function
best_parameters_summary = extract_best_parameters(DIR_RESULTS+"grid_search/")
print(best_parameters_summary)
