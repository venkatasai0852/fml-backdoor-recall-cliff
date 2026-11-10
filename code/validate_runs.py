"""
Merge validator for the four team CSVs.

Usage:
    python validate_runs.py                 # looks for ./runs_*.csv
    python validate_runs.py path/to/dir
    python validate_runs.py runs_sai.csv runs_sarang.csv ...
Exit code 0 = PASS (all hard checks), 1 = FAIL.
Prints a coverage matrix per experiment so missing cells are obvious.
"""
import sys, glob, os
import pandas as pd
import numpy as np

COLS = ['person', 'exp', 'arch', 'attack', 'rate', 'seed', 'recall', 'precision',
        'n_forget', 'method', 'detector', 'clean_acc', 'asr', 'auc',
        'tpr_at_fpr10', 'prec_at_fpr10', 'n_flags', 'n_true_in_flags',
        'time_sec', 'notes']
KEY = ['exp', 'arch', 'attack', 'rate', 'seed', 'recall', 'precision',
       'n_forget', 'method', 'detector']
E1_CONFIGS = [(1.0, 1.0), (0.95, 1.0), (0.90, 1.0), (0.75, 1.0), (0.50, 1.0),
              (0.25, 1.0), (0.90, 0.90), (0.90, 0.70), (0.50, 0.90), (0.50, 0.70)]
E1_METHODS = ['retrain_delF', 'finetune_retain', 'negrad_plus', 'scrub', 'recipe']
NO_ACC = {'poisoned', 'mia', 'relearn', 'label_acc', 'car_proxy'}
ALLOWED_EXP = {'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E7b', 'E8', 'E9', 'SISA'}

fails, warns = [], []


def fail(msg):
    fails.append(msg)
    print('FAIL:', msg)


def warn(msg):
    warns.append(msg)
    print('WARN:', msg)


def load(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            fail(f'{p}: missing columns {missing}')
            continue
        extra = [c for c in df.columns if c not in COLS]
        if extra:
            warn(f'{p}: extra columns {extra} (ignored)')
        df = df[COLS].copy()
        df['src_file'] = os.path.basename(p)
        frames.append(df)
        print(f'loaded {p}: {len(df)} rows')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


def hard_checks(df):
    for c in ['rate', 'recall', 'precision']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['clean_acc', 'asr', 'auc', 'tpr_at_fpr10', 'n_forget', 'seed']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    bad = df[~df.exp.isin(ALLOWED_EXP)]
    if len(bad):
        fail(f'unknown exp values: {sorted(bad.exp.unique())}')
    bad = df[(df.rate <= 0) | (df.rate > 1)]
    if len(bad):
        fail(f'{len(bad)} rows with rate outside (0,1]')
    bad = df[(df.recall < 0) | (df.recall > 1) | (df.precision < 0) | (df.precision > 1)]
    if len(bad):
        fail(f'{len(bad)} rows with recall/precision outside [0,1]')
    need_acc = df[~df.method.isin(NO_ACC)]
    bad = need_acc[need_acc.clean_acc.isna() | need_acc.asr.isna()]
    if len(bad):
        fail(f'{len(bad)} removal rows missing clean_acc/asr: '
             f'{bad[["exp", "method"]].drop_duplicates().to_dict("records")[:5]}')
    bad = need_acc[(need_acc.clean_acc < 0) | (need_acc.clean_acc > 100) |
                   (need_acc.asr < 0) | (need_acc.asr > 100)]
    if len(bad):
        fail(f'{len(bad)} removal rows with acc/asr outside [0,100]')
    dup = df[df.duplicated(KEY, keep=False)]
    if len(dup):
        fail(f'{len(dup)} duplicate KEY rows across all files, e.g. '
             f'{dup[KEY].head(3).to_dict("records")}')
    if df.person.nunique() < 4:
        warn(f'only persons present: {sorted(df.person.unique())}')


def has(df, exp, method=None, detector=None, seed=None, arch=None, attack=None):
    d = df[df.exp == exp]
    if method is not None:
        d = d[d.method == method]
    if detector is not None:
        d = d[d.detector == detector]
    if seed is not None:
        d = d[d.seed == seed]
    if arch is not None:
        d = d[d.arch == arch]
    if attack is not None:
        d = d[d.attack == attack]
    return len(d)


def coverage(df):
    print('\n---- coverage ----')
    for seed in [0, 1, 2]:
        d = df[(df.exp == 'E1') & (df.seed == seed) & df.method.isin(E1_METHODS)]
        cnt = d.groupby('method').size().to_dict()
        short = {m: int(cnt.get(m, 0)) for m in E1_METHODS if cnt.get(m, 0) < 10}
        print(f'E1 seed {seed}: {sum(cnt.values())}/50 method rows, per-method {cnt}')
        if short:
            warn(f'E1 seed {seed} incomplete: {short}')
        if sum(cnt.values()) > 50:
            warn(f'E1 seed {seed}: more than 50 rows (duplicates?)')
    for seed in [0, 1, 2]:
        n = has(df, 'E2', seed=seed)
        print(f'E2 seed {seed}: {n}/20 rows')
        if n < 20:
            warn(f'E2 seed {seed} incomplete')
    n = has(df, 'E3')
    print(f'E3: {n} rows (need >=10 proxies + f1_full + car_full)')
    if n < 12:
        warn('E3 incomplete')
    for (r, p) in [(0.90, 1.0), (0.50, 1.0), (0.90, 0.70), (0.50, 0.70)]:
        d = df[(df.exp == 'E4') & np.isclose(df.recall, r, atol=0.01) & np.isclose(df.precision, p, atol=0.01)]
        nv = d[d.method.str.startswith('recipe')].method.nunique()
        print(f'E4 cell (r={r},p={p}): {nv}/5 recipe variants, label_acc rows={len(d[d.method == "label_acc"])}')
        if nv < 5:
            warn(f'E4 cell (r={r},p={p}) incomplete')
    for tag in ['poisoned', 'recipe', 'gold']:
        n = has(df, 'E5', detector=tag)
        print(f'E5 {tag}: {n}/2 rows (relearn + mia)')
        if n < 2:
            warn(f'E5 {tag} incomplete')
    n = has(df, 'E6')
    print(f'E6: {n}/6 rows')
    if n < 6:
        warn('E6 incomplete')
    for arch in ['vgg16_bn', 'resnet18']:
        d = df[(df.exp == 'E7') & (df.arch == arch)]
        print(f'E7 {arch}: {len(d)} rows, attacks={sorted(d.attack.unique())}')
    n = has(df, 'E7b')
    print(f'E7b: {n} rows')
    n = has(df, 'E8')
    print(f'E8: {n}/6 rows')
    if n < 6:
        warn('E8 incomplete')
    for seed in [0, 1, 2]:
        n = has(df, 'E9', seed=seed)
        print(f'E9 seed {seed}: {n}/5 rows')
        if n < 5:
            warn(f'E9 seed {seed} incomplete')
    n = has(df, 'SISA')
    print(f'SISA: {n}/2 rows')


def main():
    args = sys.argv[1:]
    if not args:
        paths = sorted(glob.glob('runs_*.csv'))
    elif len(args) == 1 and os.path.isdir(args[0]):
        paths = sorted(glob.glob(os.path.join(args[0], 'runs_*.csv')))
    else:
        paths = args
    paths = [p for p in paths if os.path.basename(p).startswith('runs_')]
    if not paths:
        print('no runs_*.csv found')
        return 1
    df = load(paths)
    if df.empty:
        print('FAIL: no data')
        return 1
    hard_checks(df)
    coverage(df)
    print('\n==== SUMMARY ====')
    print(f'rows={len(df)} persons={sorted(df.person.unique())}')
    print(f'hard failures={len(fails)} warnings={len(warns)}')
    print('PASS' if not fails else 'FAIL')
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
