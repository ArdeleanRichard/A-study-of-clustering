import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats
import scikit_posthocs as sp
from tabulate import tabulate
import warnings

from visualization.algorithm_custom_order import algorithm_order

warnings.filterwarnings('ignore')



def create_correlation_heatmap(data, evaluation_metrics):
    """Create correlation heatmap between metrics."""
    # Select metrics
    # metrics = [m for m in evaluation_metrics if m != 'norm_davies_bouldin_score']

    corr_data = data[evaluation_metrics].dropna()

    # Calculate correlation matrix
    corr_matrix = corr_data.corr(method='spearman')

    # Create heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5, annot=True, fmt='.2f')

    plt.title('Spearman Correlation between Evaluation Metrics')
    plt.tight_layout()
    plt.savefig("./correlation_of_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Wrote correlation heatmap: ./correlation_of_metrics.png")

    return corr_matrix


# Calculate failure rates
def calculate_failure_rates(data, evaluation_metrics):
    """Calculate the percentage of datasets each algorithm fails on."""
    failures = {}
    algorithms = data['algorithm'].unique()

    # for algo in algorithms:
    #     algo_data = data[data['algorithm'] == algo]
    #     total_datasets = len(algo_data)
    #
    #     # Count datasets where algorithm failed (-1 values)
    #     failures_count = sum((algo_data['adjusted_rand_score'] == -1) |
    #                          (algo_data['adjusted_rand_score'].isna()))
    #
    #     failures[algo] = (failures_count / total_datasets) * 100
    for algo in algorithms:
        sub = data[data['algorithm'] == algo]
        # count rows where every metric is NaN
        failed = sub[evaluation_metrics].isna().all(axis=1).sum()
        failures[algo] = failed / len(sub) * 100


    return pd.Series(failures)


def run_friedman_tests(data, metrics, threshold_evaluated_datasets=2, threshold_evaluated_algorithms_percentage=0.5):
    """Run Friedman tests for each metric and return detailed results."""
    results = {}

    for metric in metrics:
        # Create pivot table: datasets as rows, algorithms as columns
        pivot = data.pivot_table(index='dataset', columns='algorithm', values=metric)

        # Count valid datasets (rows with at least 2 non-NaN values)
        valid_rows = pivot.dropna(thresh=threshold_evaluated_datasets)

        if len(valid_rows) < 5:  # Need at least a few datasets
            results[metric] = {
                'statistic': None,
                'p_value': None,
                'significant': False,
                'valid_datasets': len(valid_rows),
                'valid_algorithms': 0
            }
            continue

        # Extract algorithms with sufficient data
        valid_algos = []
        for algo in pivot.columns:
            if valid_rows[algo].count() >= len(valid_rows) * threshold_evaluated_algorithms_percentage:  # Algorithm has results for at least 50% of valid datasets
                valid_algos.append(algo)

        if len(valid_algos) < 2:  # Need at least 2 algorithms
            results[metric] = {
                'statistic': None,
                'p_value': None,
                'significant': False,
                'valid_datasets': len(valid_rows),
                'valid_algorithms': len(valid_algos)
            }
            continue

        # Filter to valid data
        valid_data = valid_rows[valid_algos].copy()

        # Handle NaN values
        if metric in ['davies_bouldin_score']:
            # For metrics where lower is better
            max_vals = valid_data.max()
            for algo in valid_data.columns:
                valid_data[algo] = valid_data[algo].fillna(max_vals[algo] * 1.5)
        else:
            # For metrics where higher is better
            min_vals = valid_data.min()
            for algo in valid_data.columns:
                valid_data[algo] = valid_data[algo].fillna(min_vals[algo] * 0.5)

        try:
            # Perform Friedman test
            statistic, p_value = stats.friedmanchisquare(*[valid_data[algo] for algo in valid_algos])

            results[metric] = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'valid_datasets': len(valid_rows),
                'valid_algorithms': len(valid_algos)
            }
        except Exception as e:
            results[metric] = {
                'statistic': None,
                'p_value': None,
                'significant': False,
                'valid_datasets': len(valid_rows),
                'valid_algorithms': 0,
                'error': str(e)
            }

    return results


def calculate_ranks_with_significance(data, metrics):
    """Calculate ranks and perform pairwise significance tests."""
    # Calculate failure rates
    failure_rates = calculate_failure_rates(data, metrics)

    all_ranks = {}
    all_pvalues = {}

    for metric in metrics:
        # Create pivot table
        pivot = data.pivot_table(index='dataset', columns='algorithm', values=metric)

        # Handle NaN values
        if metric in ['davies_bouldin_score']:
            # For metrics where lower is better
            ranked = pivot.rank(axis=1, ascending=True, na_option='bottom')
        else:
            # For metrics where higher is better
            ranked = pivot.rank(axis=1, ascending=False, na_option='bottom')

        # Calculate average rank
        avg_ranks = ranked.mean()
        all_ranks[metric] = avg_ranks

        # Check if we have enough valid data for pairwise tests
        valid_rows = pivot.dropna(thresh=2)
        if len(valid_rows) < 5:
            continue

        # Extract algorithms with sufficient data
        valid_algos = []
        for algo in pivot.columns:
            if valid_rows[algo].count() >= len(valid_rows) * 0.5:
                valid_algos.append(algo)

        if len(valid_algos) < 2:
            continue

        # Filter to valid data
        valid_data = valid_rows[valid_algos].copy()

        # Handle NaN values
        if metric in ['davies_bouldin_score']:
            max_vals = valid_data.max()
            for algo in valid_data.columns:
                valid_data[algo] = valid_data[algo].fillna(max_vals[algo] * 1.5)
        else:
            min_vals = valid_data.min()
            for algo in valid_data.columns:
                valid_data[algo] = valid_data[algo].fillna(min_vals[algo] * 0.5)

        try:
            # FIX: Create a matrix for nemenyi test with proper dimensions
            data_array = np.array(valid_data)

            # Compute the Nemenyi post-hoc test manually if scikit_posthocs has issues
            # First compute ranks on each row
            if metric in ['davies_bouldin_score']:
                ranks = np.zeros_like(data_array)
                for i in range(data_array.shape[0]):  # for each dataset
                    ranks[i, :] = stats.rankdata(data_array[i, :], method='average')
            else:
                ranks = np.zeros_like(data_array)
                for i in range(data_array.shape[0]):  # for each dataset
                    ranks[i, :] = stats.rankdata(-data_array[i, :], method='average')

            # Calculate mean ranks
            mean_ranks = np.mean(ranks, axis=0)

            # Number of algorithms and datasets
            k = len(valid_algos)
            n = len(valid_rows)

            # Critical difference for Nemenyi test at alpha=0.05
            critical_diff = np.sqrt((k * (k + 1)) / (6 * n)) * 1.96  # approximate for alpha=0.05

            # Create pairwise p-value matrix
            pvalue_matrix = np.ones((k, k))
            for i in range(k):
                for j in range(i + 1, k):
                    # Calculate z-score
                    z = abs(mean_ranks[i] - mean_ranks[j]) / np.sqrt((k * (k + 1)) / (6 * n))
                    # Convert to p-value (two-tailed test)
                    p = 2 * (1 - stats.norm.cdf(z))
                    pvalue_matrix[i, j] = p
                    pvalue_matrix[j, i] = p

            posthoc_df = pd.DataFrame(pvalue_matrix, index=valid_algos, columns=valid_algos)
            all_pvalues[metric] = posthoc_df

        except Exception as e:
            print(f"Error in post-hoc test for {metric}: {e}")

    # Calculate overall rank (average across all metrics)
    rank_df = pd.DataFrame(all_ranks)
    rank_df['overall'] = rank_df.mean(axis=1)

    # Add failure rates
    rank_df['failure_rate'] = failure_rates

    # Sort by overall rank
    rank_df = rank_df.sort_values('overall')

    for metric, df in all_pvalues.items():
        df.to_csv(f"./pvalues_{metric}.csv")
        print(f"Wrote p-value CSV: ./pvalues_{metric}.csv")

    return rank_df, all_pvalues


def create_pvalue_heatmap(pvalues, metric):
    """Create heatmap showing pairwise p-values between algorithms."""
    if metric not in pvalues:
        return None

    p_matrix = pvalues[metric]

    # Create figure with size adaptive to number of algorithms
    fig_size = max(8, len(p_matrix) * 0.5)
    plt.figure(figsize=(fig_size, fig_size))

    # Create mask for diagonal only (so all pairwise comparisons are visible)
    mask = np.zeros_like(p_matrix, dtype=bool)
    np.fill_diagonal(mask, True)  # Only hide diagonal

    # Create heatmap with modified color scheme
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    heatmap = sns.heatmap(p_matrix, mask=mask, cmap=cmap, vmax=0.05, vmin=0, center=0.025,
                          square=True, linewidths=.5, cbar_kws={"shrink": .5},
                          annot=True, fmt='.3f')

    plt.title(f'Pairwise p-values for {metric}\n(p-values < 0.05 indicate significant differences)')
    plt.tight_layout()
    plt.savefig(f"./pvalue_{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote p-value heatmap: ./pvalue_{metric}.png")



def reorder_matrix(p_matrix):
    # keep only those in the matrix, in your order
    ordered = [a for a in algorithm_order if a in p_matrix.index]
    # then append any algorithms missing from your list
    remainder = [a for a in p_matrix.index if a not in ordered]
    final_order = ordered + remainder
    return p_matrix.reindex(index=final_order, columns=final_order)

def create_pvalue_heatmap_only_significance(pvalues, metric, order=True):
    """Discrete significance‑level heatmap for metric."""
    if metric not in pvalues:
        return

    if order == True:
        p = reorder_matrix(pvalues[metric])
        suffix = "_ordered"
    else:
        p = pvalues[metric]
        suffix = ""

    # map to symbols
    sym = p.copy().applymap(
        lambda pv: '**' if pv < 0.01
                   else '*'  if pv < 0.05
                   else ''
    )
    # numeric codes for colors
    code = sym.replace({'': 0, '*': 1, '**': 2})

    # define a three‑color palette: e.g. grey, lightblue, blue
    cmap = ListedColormap(['lightgray', 'skyblue', 'steelblue'])

    size = max(8, len(p) * 0.5)
    plt.figure(figsize=(size, size))
    sns.heatmap(
        code,
        cmap=cmap,
        cbar=False,
        annot=sym,
        fmt='',
        square=True,
        linewidths=.5,
        linecolor='white'
    )
    plt.title(f'{metric}: discrete significance levels')
    plt.tight_layout()
    plt.savefig(f"./pvalue_signif_{metric}{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote discrete‑level heatmap: ./pvalue_signif_{metric}{suffix}.png")


def plot_significance_counts(pvalues, metric, order=True):
    if metric not in pvalues:
        return

    if order == True:
        p = reorder_matrix(pvalues[metric])
    else:
        p = pvalues[metric]

    # binary matrix: 1 if p<0.05, else 0 (including diagonal)
    sig_binary = (p < 0.05).astype(int)
    # each row’s sum = number of other algos it differs from
    counts = sig_binary.sum(axis=1)

    # plot
    plt.figure(figsize=(max(6,len(counts)*0.25), 4))
    counts.plot.bar()
    plt.xticks(rotation=90)
    plt.ylabel("Count of p<0.05 vs others")
    plt.title(f"{metric}: number of significant differences")
    plt.tight_layout()
    fn = f"./barcount_signif_{metric}.png"
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote bar‐plot: {fn}")

def create_significance_table(evaluation_metrics, friedman_results):
    """Create table showing Friedman test results for each metric."""
    table_data = []

    for metric in evaluation_metrics:
        if metric in friedman_results:
            result = friedman_results[metric]

            if result['statistic'] is not None:
                p_value_str = f"{result['p_value']:.3e}" if result['p_value'] < 0.0001 else f"{result['p_value']:.6f}"

            row = [
                metric,
                result['valid_datasets'],
                result['valid_algorithms'],
                f"{result['statistic']:.2f}" if result['statistic'] is not None else "N/A",
                p_value_str if result['statistic'] is not None else "N/A",
                "Yes" if result['significant'] else "No" if result['statistic'] is not None else "N/A",
            ]

        else:
            row = [metric, "N/A", "N/A", "N/A", "N/A", "N/A"]

        table_data.append(row)

    # Create DataFrame
    significance_table = pd.DataFrame(
        table_data,
        columns=['Metric', 'Valid Datasets', 'Valid Algorithms', 'Friedman Statistic', 'p-value', 'Significant (p<0.05)']
    )

    return significance_table



def format_comprehensive_table(evaluation_metrics, ranks, friedman_results):
    """Create a comprehensive table with all algorithms, ranks, failure rates, and significance."""
    # Start with the rank dataframe
    table = ranks.copy()

    # Add rank column
    table = table.reset_index().rename(columns={'index': 'algorithm'})
    table.insert(0, 'rank', range(1, len(table) + 1))
    table = table.set_index('rank')

    # Add significance indicators to column names
    new_columns = {}
    for metric in evaluation_metrics:
        if metric in friedman_results and friedman_results[metric]['significant']:
            # Format p-value
            p_value = friedman_results[metric]['p_value']
            significance = ''
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'

            new_columns[metric] = f"{metric} {significance}"
        else:
            new_columns[metric] = metric

    # Keep other columns unchanged
    for col in table.columns:
        if col not in evaluation_metrics and col != 'algorithm':
            new_columns[col] = col

    # Rename columns
    table = table.rename(columns=new_columns)

    # Format numeric columns to 2 decimal places
    for col in table.columns:
        if col not in ['algorithm', 'failure_rate']:  # Keep failure rate with more precision
            table[col] = table[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

    # Format failure rate with % symbol
    table['failure_rate'] = table['failure_rate'].map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")

    return table





def format_value(val):
    # If numeric and integer-like, show as int; else preserve original
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        else:
            return str(f)
    except:
        return str(val)


def escape_underscore(text: str) -> str:
    return text.replace('_', r'\_')


def datafrane_to_latex(df,
                 output_path: str,
                 caption: str,
                 label: str):
    cols = df.columns.tolist()
    esc_cols = [escape_underscore(c) for c in cols]
    ncol = len(cols)
    col_spec = '|' + '|'.join(['l'] * ncol) + '|'

    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(f'\\caption{{{escape_underscore(caption)}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append(fr'\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\hline')
    header = ' & '.join(esc_cols) + r' \\'
    lines.append(header)
    lines.append(r'\hline')
    for _, row in df.iterrows():
        vals = [escape_underscore(format_value(row[c])) for c in cols]
        lines.append(' & '.join(vals) + r' \\')
        lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Wrote: {output_path}')


def complete_analysis():
    """Execute complete analysis with all requested components."""
    df = pd.read_csv('../results/results.csv')
    df_clean = df.replace(-1, np.nan)

    evaluation_metrics = ['adjusted_rand_score', 'adjusted_mutual_info_score', 'purity_score', 'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'norm_davies_bouldin_score']

    print("\nCreate correlation heatmap - Analyzing correlation between metrics...")
    corr_matrix = create_correlation_heatmap(df_clean, evaluation_metrics)

    print("\nPerforming Friedman tests for each metric...")
    friedman_results = run_friedman_tests(df_clean, evaluation_metrics, 24, 0.9)

    print("\nCreating table of statistical significance...")
    significance_table = create_significance_table(evaluation_metrics, friedman_results)
    datafrane_to_latex(significance_table, "./table_significance.tex", caption="Friedman Results", label="friedman_significance")

    print("\nCalculating algorithm rankings and failure rates...")
    ranks, pvalues = calculate_ranks_with_significance(df_clean, evaluation_metrics)
    full_table = format_comprehensive_table(evaluation_metrics, ranks, friedman_results)
    datafrane_to_latex(full_table, "./table_ranks_full.tex", caption="Rank Results (* p<0.05, ** p<0.01, *** p<0.001 indicate statistical significance)", label="rank_results")

    print("\nCreating p-value heatmaps for metrics with significant differences...")
    for metric in evaluation_metrics:
        if metric in friedman_results and friedman_results[metric]['significant']:
            # create_pvalue_heatmap(pvalues, metric)
            # create_pvalue_heatmap_only_significance(pvalues, metric, order=False)
            create_pvalue_heatmap_only_significance(pvalues, metric)
            # plot_significance_counts(pvalues, metric)


    return {
        'friedman_results': friedman_results,
        'ranks': ranks,
        'pvalues': pvalues,
        'full_table': full_table,
        'significance_table': significance_table,
        'correlation_matrix': corr_matrix,
    }


if __name__ == "__main__":
    results = complete_analysis()