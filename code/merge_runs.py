"""
Merge the four team CSVs into one, after validate_runs.py passes.

Usage:
    python merge_runs.py                 # ./runs_*.csv -> merged_runs.csv
    python merge_runs.py dir_in out.csv
Prints per-experiment/per-person counts and the E1 grid completeness.
"""
import sys, glob, os
import pandas as pd

COLS = ['person', 'exp', 'arch', 'attack', 'rate', 'seed', 'recall', 'precision',
        'n_forget', 'method', 'detector', 'clean_acc', 'asr', 'auc',
        'tpr_at_fpr10', 'prec_at_fpr10', 'n_flags', 'n_true_in_flags',
        'time_sec', 'notes']


def main():
    din = sys.argv[1] if len(sys.argv) > 1 else '.'
    out = sys.argv[2] if len(sys.argv) > 2 else 'merged_runs.csv'
    paths = [p for p in sorted(glob.glob(os.path.join(din, 'runs_*.csv')))
             if 'merged' not in os.path.basename(p)]
    if not paths:
        print('no runs_*.csv in', din)
        return 1
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for c in COLS:
        if c not in df.columns:
            df[c] = ''
    df = df[COLS]
    df.to_csv(out, index=False)
    print(f'merged {len(paths)} files -> {out} ({len(df)} rows)')
    print(df.groupby(['person', 'exp']).size().to_string())
    e1 = df[(df.exp == 'E1') & df.method.isin(
        ['retrain_delF', 'finetune_retain', 'negrad_plus', 'scrub', 'recipe'])]
    piv = e1.pivot_table(index=['seed'], columns='method', values='asr', aggfunc='count')
    print('\nE1 row counts per seed/method:')
    print(piv.to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
