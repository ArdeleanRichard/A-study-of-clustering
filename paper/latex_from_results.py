import os
import pandas as pd

from latex_utils import csv_to_latex, wrap_and_relabel

RENAME_MAP = {
    'adjusted_rand_score': 'ARI',
    'adjusted_mutual_info_score': 'AMI',
    'purity_score':           'Purity',
    'silhouette_score':       'SS',
    'calinski_harabasz_score':'CHS',
    'davies_bouldin_score':   'DBS',
    'norm_davies_bouldin_score':'NDBS',
    'norm_calinski_harabasz_score':'NCHS',
}


def split_csv_by_dataset(input_csv: str, output_dir: str):
    df = pd.read_csv(input_csv)
    df = df.rename(columns=RENAME_MAP)
    os.makedirs(output_dir, exist_ok=True)

    for dset in df['dataset'].unique():
        sub = df[df['dataset'] == dset]
        non_empty = [c for c in sub.columns
                     if sub[c].apply(lambda x: str(x).strip() not in ['', 'nan']).any()]
        sub_clean = sub[non_empty]
        out_path = os.path.join(output_dir, f"{dset}.csv")
        if 'dataset' in sub_clean.columns:
            sub_clean = sub_clean.drop(['dataset'], axis=1)
        sub_clean.to_csv(out_path, index=False)
        print(f"Saved: {out_path} (cols: {non_empty})")


if __name__ == '__main__':
    prefix = "results_all_metrics"
    INPUT_CSV = f'../results/{prefix}.csv'
    OUTPUT_DIR = f'../paper/{prefix}_split_csvs'
    split_csv_by_dataset(INPUT_CSV, OUTPUT_DIR)

    INPUT_DIR = f'../paper/{prefix}_split_csvs/'
    OUTPUT_DIR = f'../paper/{prefix}_latex_tables'
    CAPTION_PREFIX = 'Results on dataset'
    LABEL_PREFIX = 'tab:params'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.endswith('.csv'): continue
        name = os.path.splitext(fname)[0]
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, f'{name}.tex')
        caption = f"{CAPTION_PREFIX} {name}"  # adjust if needed
        label = f"{LABEL_PREFIX}:{name}"
        csv_to_latex(in_path, out_path, caption, label)

    TEX_DIR = f'../paper/{prefix}_latex_tables'
    OUTPUT_FILE = f'../paper/{prefix}_all_tables.tex'
    NEW_LABEL_BASE = 'S'
    wrap_and_relabel(TEX_DIR, OUTPUT_FILE, NEW_LABEL_BASE, start_label=50)

