import numpy as np
import pandas as pd
from rank_aggregation.rankagg import FullListRankAggregator

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

ranking_matrix = []
for met in METRICS:
    ranks_list = []
    ranks_dict = []
    for dset in datasets:
        sub_df = df[df['dataset'] == dset]
        sub_df = sub_df.dropna(subset=[met])
        ranked = sub_df.sort_values(by=met, ascending=False)
        rank_order = ranked['algorithm'].tolist()
        ranks_list.append(rank_order)

        # Create rank dictionary: method -> score
        rank_score = dict(zip(sub_df['algorithm'], sub_df[met]))
        ranks_dict.append(rank_score)

    aggregator = FullListRankAggregator()
    borda_result = aggregator.aggregate_ranks(ranks_dict, method='borda')
    print(f"Borda Ranking for {met}:")
    print(borda_result[1])


    sorted_borda = sorted(borda_result[1].items(), key=lambda x: x[1])

    sorted_methods = [method for method, score in sorted_borda]
    print(sorted_methods)

    ranking_matrix.append(sorted_methods)

    print()


# Create DataFrame with rows as algorithms and columns as metrics
ranking_df = pd.DataFrame(np.array(ranking_matrix).T, columns=METRICS, index=list(range(1, len(methods)+1)))

# Save to CSV
ranking_df.to_csv("aggregated_ranks.csv")

print("Aggregated rankings have been saved to 'aggregated_ranks.csv'.")


