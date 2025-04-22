import os
import pandas as pd

from latex_utils import csv_to_latex, wrap_and_relabel


if __name__ == '__main__':
    prefix = "aggregated_ranks"
    INPUT_CSV = f'../statistics/ranking/{prefix}.csv'
    OUTPUT_DIR = f'../statistics/ranking/'

    CAPTION_PREFIX = 'Borda ranking by performance metrics'
    LABEL_PREFIX = 'tab:'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    caption = f"{CAPTION_PREFIX}"  # adjust if needed
    label = f"{LABEL_PREFIX}:borda"
    out_path = os.path.join(OUTPUT_DIR, f'{prefix}.tex')
    csv_to_latex(INPUT_CSV, out_path, caption, label)

    TEX_DIR = f'../statistics/ranking/{prefix}_latex_tables'
    OUTPUT_FILE = f'../statistics/ranking/{prefix}_all_tables.tex'
    NEW_LABEL_BASE = 'S'
    wrap_and_relabel(TEX_DIR, OUTPUT_FILE, NEW_LABEL_BASE, start_label=50)

