import os

import numpy as np
import pandas as pd

from constants import DIR_RESULTS

def normalize_column(df, column_name):
    log = np.log1p(2+df[column_name])
    return log / log.max()

# Function to compute priority score
def compute_priority_score(row, df):
    # Example weighted combination of metrics
    # print(row['calinski_harabasz_score'])
    # print(row['davies_bouldin_score'])
    normalized_calinski_harabasz  = normalize_column(df, 'calinski_harabasz_score')[row.name]

    return (
        0.1 * row['silhouette_score'] +
        0.1 * normalized_calinski_harabasz +
        0.1 * row['norm_davies_bouldin_score'] +
        0.3 * row['adjusted_rand_score'] +
        0.3 * row['adjusted_mutual_info_score'] +
        0.1 * row['purity_score']
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
            # Drop rows with all -1 in the required scores -> all parameter combos gave 1 cluster
            df_filtered = df.loc[~(df[required_scores] == -1).all(axis=1)]

            # Check if the resulting DataFrame is empty, means that no actual scores exists, however norm_dbs is inf
            if df_filtered.empty:
                df['norm_davies_bouldin_score'] = -1
            else:
                df = df_filtered

            # Compute priority score for each row
            df['priority_score'] = df.apply(compute_priority_score, axis=1, df=df)
            # Find the row with the best priority score
            best_row = df.loc[df['priority_score'].idxmax()]


            best_params = {
                "algorithm": algorithm,
                "dataset": dataset,
                **best_row.to_dict(),  # Include all parameters and scores
            }
            #print(best_row)
            summary.append(best_params)

    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary)

    # Save the summary to a new CSV file
    summary_df.to_csv(DIR_RESULTS+"grid_search_summary.csv", index=False)

    return summary_df

# Run the function
best_parameters_summary = extract_best_parameters(DIR_RESULTS+"grid_search/test_norm/")
print(best_parameters_summary)
