import os
import re

import pandas as pd

def format_value(val):
    # If numeric and integer-like, show as int; else preserve original
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        else:
            return f"{f:.2f}"  # str(f)
    except:
        return str(val)


def escape_underscore(text: str) -> str:
    return text.replace('_', r'\_')


def csv_to_latex(csv_path: str,
                 output_path: str,
                 caption: str,
                 label: str):
    df = pd.read_csv(csv_path)
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



def wrap_and_relabel(tex_dir: str,
                     output_file: str,
                     new_label_base: str,
                     start_label: int = 1):
    tex_files = sorted(
        f for f in os.listdir(tex_dir) if f.lower().endswith('.tex')
    )
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    with open(output_file, 'w') as out:
        out.write('% Automatically generated master table file\n\n')
        for idx, fname in enumerate(tex_files, start=start_label):
            path = os.path.join(tex_dir, fname)
            with open(path, 'r') as f:
                content = f.read()

            # Create new sequential label, e.g. S1_Table
            new_label = f"{new_label_base}{idx}_Table"
            # Replace the first \label{...} occurrence with new label
            relabeled = re.sub(
                r'\\label\{.*?\}',
                fr'\\label{{{new_label}}}',
                content,
                count=1
            )

            out.write(relabeled + '\n\n')#\\clearpage\n\n')

    print(f'Combined {len(tex_files)} tables into {output_file} ' \
          f'with sequential labels starting at {new_label_base}1_Table')