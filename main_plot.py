import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib


from constants import DIR_RESULTS
from gs_datasets import load_data_simple, load_data_hd, load_data_nonconvex, load_data_overlap_and_imbalance, load_data_imbalance, load_data_overlap
from constants import DIR_FIGURES

def normalize_chs(df):
    df['norm_calinski_harabasz_score'] = np.log1p(1+df['calinski_harabasz_score'])
    df['norm_calinski_harabasz_score'] = df['norm_calinski_harabasz_score'] / df['norm_calinski_harabasz_score'].max()
    return df

def transform_data_np(df, algorithms, datasets, scores, data_names):
    score_columns = [
        "adjusted_rand_score",
        "adjusted_mutual_info_score",
        "purity_score",
        "silhouette_score",
        "norm_calinski_harabasz_score",
        "norm_davies_bouldin_score"
    ]

    # Initialize an array to store the scores
    data = np.zeros((len(algorithms), len(data_names), len(score_columns)))

    # Populate the array and normalize per dataset
    for j, dataset in enumerate(data_names):
        # Filter rows for the current dataset
        dataset_rows = df[df["dataset"] == dataset]

        # Normalize scores for the dataset independently
        dataset_normalized = dataset_rows.copy()
        dataset_normalized[f"norm_calinski_harabasz_score"] = np.log1p(1+dataset_rows["calinski_harabasz_score"])  # Log normalization
        dataset_normalized[f"norm_calinski_harabasz_score"] = dataset_normalized[f"norm_calinski_harabasz_score"] / dataset_normalized[f"norm_calinski_harabasz_score"].max()  # Min-max scaling

        # Replace score columns with normalized versions
        score_columns_norm = [
            "adjusted_rand_score",
            "adjusted_mutual_info_score",
            "purity_score",
            "silhouette_score",
            "norm_calinski_harabasz_score",
            "norm_davies_bouldin_score"
        ]

        # Update the normalized scores in the dataframe
        for i, algorithm in enumerate(algorithms):
            # Filter rows for the current algorithm
            row = dataset_normalized[dataset_normalized["algorithm"] == algorithm]
            if not row.empty:
                # Extract the normalized scores and store them in the array
                data[i, j, :] = row[score_columns_norm].values[0]

    return data


def plot_hierarchical_visualization(title, df, data_names):
    algorithms = df["algorithm"].unique()
    datasets = df["dataset"].unique()
    scores = ["ARI", "AMI", "Pur", "SS", "CHS", "DBS"]

    # Parameters
    num_algorithms = len(algorithms)
    num_datasets = len(data_names)
    num_scores = len(scores)

    # data (shape: algorithms x datasets x scores)
    data = transform_data_np(df, algorithms, datasets, scores, data_names)

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.set_dpi(600)

    # Overall grid dimensions, with spacing
    total_width = 1.0
    total_height = 1.0

    algo_height = total_height / (num_algorithms + num_algorithms - 1)  # Include spacing rows
    dataset_width = total_width / (num_datasets + (num_datasets - 1) * (1 / num_scores))  # Smaller spacing columns

    spacing_width = dataset_width / num_scores  # Spacing size equivalent to one score width

    # Loop through algorithms and datasets
    for algo_idx in range(num_algorithms):
        for dataset_idx in range(num_datasets):
            # Top-left corner of the dataset block
            x0 = dataset_width * dataset_idx + spacing_width * dataset_idx
            y0 = total_height - algo_height * (2 * algo_idx + 1)

            # Draw the dataset block
            ax.add_patch(Rectangle((x0, y0), dataset_width, algo_height, edgecolor='black', facecolor='none'))

            # Draw small rectangles for validation scores inside the dataset block
            score_width = dataset_width / num_scores
            for score_idx in range(num_scores):
                score_x0 = x0 + score_width * score_idx
                score_y0 = y0

                # Map the score value to a color
                color = plt.cm.Greens(data[algo_idx, dataset_idx, score_idx])
                ax.add_patch(Rectangle((score_x0, score_y0), score_width, algo_height, color=color))

    # Add labels and formatting
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.set_xticks([((dataset_width + spacing_width) * i + 0.5 * dataset_width) for i in range(num_datasets)])
    ax.set_xticklabels(data_names)
    ax.set_yticks([(total_height - (2 * i + 0.5) * algo_height) for i in range(num_algorithms)])  # Align ticks with rows
    ax.set_yticklabels(algorithms)
    ax.set_title(title)
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Algorithms")



    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="Greens", norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Validation Score")



    # Add the score ticks on top of the colored squares
    # Define positions for the top ticks, above the colored blocks
    top_ticks_x = []
    top_tick_labels = []
    for i in range(num_datasets):
        for score_idx in range(num_scores):
            top_ticks_x.append((dataset_width + spacing_width) * i + score_idx * (dataset_width / num_scores) + 0.5 * (
                        dataset_width / num_scores))
        top_tick_labels.extend(scores)



    # Create a second x-axis on top with different labels (if done before colorbar, it moves it)
    ax2 = ax.twiny()
    ax2.set_xticks(top_ticks_x)
    ax2.set_xticklabels(top_tick_labels)

    # Adjust the position of the second x-axis
    ax2.xaxis.set_ticks_position('top')

    plt.tight_layout()
    # plt.show()
    plt.savefig(DIR_FIGURES + f"{title}.png")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(DIR_RESULTS + 'grid_search_summary.csv')

    datasets = load_data_simple()
    data_names = [x[0] for x in datasets]
    plot_hierarchical_visualization(title="Comparison on simple datasets", df=df, data_names=data_names)

    datasets = load_data_overlap()
    data_names = [x[0] for x in datasets]
    plot_hierarchical_visualization(title="Comparison on overlap datasets", df=df, data_names=data_names)

    datasets = load_data_imbalance()
    data_names = [x[0] for x in datasets]
    plot_hierarchical_visualization(title="Comparison on imbalance datasets", df=df, data_names=data_names)

    datasets = load_data_overlap_and_imbalance()
    data_names = [x[0] for x in datasets]
    plot_hierarchical_visualization(title="Comparison on overlap and imbalance datasets", df=df, data_names=data_names)

    datasets = load_data_nonconvex()
    data_names = [x[0] for x in datasets]
    plot_hierarchical_visualization(title="Comparison on nonconvex datasets", df=df, data_names=data_names)

    datasets = load_data_hd()
    data_names = [x[0] for x in datasets]
    plot_hierarchical_visualization(title="Comparison on high dimensional datasets", df=df, data_names=data_names)