import os
import pandas as pd

from results.latex_utils import csv_to_latex, wrap_and_relabel


def split_csv_by_algorithm(input_csv: str, output_dir: str):
    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    print(len(df['algorithm'].unique()))
    print(df['algorithm'].unique())
    for alg in df['algorithm'].unique():
        sub = df[df['algorithm'] == alg]
        # keep only columns with at least one non-null, non-blank entry
        non_empty_cols = [col for col in sub.columns
                          if sub[col].astype(str).str.strip().replace('nan','').replace('','').any()]
        sub_clean = sub[non_empty_cols]
        out_path = os.path.join(output_dir, f"{alg}.csv")
        sub_clean.to_csv(out_path, index=False)
        print(f"Saved: {out_path} (columns: {non_empty_cols})")



if __name__ == '__main__':
    prefix = "best_params"
    INPUT_CSV = f'../paper/{prefix}.csv'
    OUTPUT_DIR = f'../paper/{prefix}_split_csvs'
    split_csv_by_algorithm(INPUT_CSV, OUTPUT_DIR)

    INPUT_DIR = f'../paper/{prefix}_split_csvs/'
    OUTPUT_DIR = f'../paper/{prefix}_latex_tables'
    CAPTION_PREFIX = 'Best params for'
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
    wrap_and_relabel(TEX_DIR, OUTPUT_FILE, NEW_LABEL_BASE)

