import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

df = pd.read_csv("./results/time_analysis_samples.csv")
df["samples"] = df["dataset"].str.extract(r"D3_(\d+)").astype(int)
score_cols = [
    "adjusted_rand_score", "adjusted_mutual_info_score",
    "purity_score", "silhouette_score",
    "calinski_harabasz_score", "davies_bouldin_score"
]
df["is_invalid"] = df[score_cols].eq(-1).all(axis=1)

# 2. Compute each algorithm’s max execution time
algo_max = df.groupby("algorithm")["execution_time"].max()

# 3. Split algorithms into fast (<10s) vs slow (>=10s)
fast_algos = algo_max[algo_max < 10].index
slow_algos = algo_max[algo_max >= 10].index

fast_df = df[df["algorithm"].isin(fast_algos)].copy()
slow_df = df[df["algorithm"].isin(slow_algos)].copy()

# 4. Bin the fast algorithms by their max exec time quartiles
quantiles = algo_max.loc[fast_algos].quantile([0, .25, .5, .75, 1.0]).values
edges = np.unique(quantiles)
labels = [f"{edges[i]:.3f}–{edges[i+1]:.3f}s" for i in range(len(edges)-1)]

fast_max = algo_max.loc[fast_algos]
fast_bins = pd.cut(fast_max, bins=edges, labels=labels, include_lowest=True)
algo_to_fastbin = fast_bins.to_dict()
fast_df["max_time_bin"] = fast_df["algorithm"].map(algo_to_fastbin)

# 5. Plot helper
def plot_group(subdf, title, save_file):
    plt.figure(figsize=(12, 6))

    algos = sorted(subdf["algorithm"].unique())

    M = 10
    colors = sns.color_palette("colorblind", n_colors=M)
    markers    = ['o']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

    plt.rc('axes', prop_cycle=cycler('linestyle', linestyles) * cycler('color', colors) * cycler('marker', markers))

    plt.scatter([], [], marker='x', c='red', s=100, label="Invalid run")

    for algo in algos:
        d = subdf[subdf["algorithm"] == algo].sort_values("samples")
        valid = d
        plt.plot(valid["samples"], valid["execution_time"],
                 label=algo, alpha=0.8)

        inval = d[d["is_invalid"]]
        if not inval.empty:
            plt.scatter(inval["samples"], inval["execution_time"],
                        marker='x', c='red', s=100)

    plt.title(title)
    plt.xlabel("Number of Samples")
    plt.ylabel("Execution Time (s)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    plt.tight_layout()
    plt.savefig(f"./paper/time_samples/{save_file}.png")
    plt.close()


for id, (bin_label, group) in enumerate(fast_df.groupby("max_time_bin")):
    plot_group(group, f"Fast Algorithms (exec. time {bin_label})", f"time_sam_fast_{id}")

if not slow_df.empty:
    plot_group(slow_df, "Slow Algorithms (exec. time ≥ 10s)", f"time_sam_slow")