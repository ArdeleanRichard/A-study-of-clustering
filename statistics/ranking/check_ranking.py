import pandas as pd

CSV_FILE = "../results.csv"

# List your metrics in the same order you want to rank
METRICS = [
    "adjusted_rand_score",
    "adjusted_mutual_info_score",
    "purity_score",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
]

df = pd.read_csv(CSV_FILE)
datasets = df['dataset'].unique()
methods = df['algorithm'].unique()

def check(data, selected_algorithm="acedec", selected_metric="adjusted_rand_score"):
    filtered_data = data[data['algorithm'] == selected_algorithm]
    metric_performance = filtered_data[selected_metric].tolist()

    return metric_performance

birch = check(df, selected_algorithm="birch", selected_metric="adjusted_rand_score")
spectral = check(df, selected_algorithm="spectral", selected_metric="adjusted_rand_score")
print(birch)
print(spectral)

print(len(birch))
print(len(spectral))

print(sum(birch)/len(birch))
print(sum(spectral)/len(spectral))


def compute_borda_ranking(df, selected_metric):
    borda_scores = {}
    datasets = df['dataset'].unique()

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        sorted_algorithms = dataset_df.sort_values(by=selected_metric, ascending=False)

        for rank, algorithm in enumerate(sorted_algorithms['algorithm'], start=1):
            if algorithm not in borda_scores:
                borda_scores[algorithm] = 0
            borda_scores[algorithm] += rank  # Add the rank to the algorithm's Borda score

    borda_ranking = pd.DataFrame(list(borda_scores.items()), columns=['algorithm', 'total_rank'])

    borda_ranking = borda_ranking.sort_values(by='total_rank')

    return borda_ranking


# Example usage: Compute Borda ranks for 'adjusted_rand_score'
borda_ranking = compute_borda_ranking(df, selected_metric="adjusted_rand_score")

# Display the Borda ranking
print(borda_ranking)