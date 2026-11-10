"""Generate FML_RF_Removal_Study.ipynb -- a ~1 h ECE/RF removal study.

Scales the validated 3-min RadioML pilot into a 3-seed study that mirrors the
CIFAR recall/precision thesis in RF automatic modulation classification and
sketches the P1 headline quantity gamma* (detector recall required for defence
success ASR<=5%, as a function of SNR).

Standalone: does NOT embed the CIFAR frozen core. Reuses the pilot's loader,
1-D CNN, I/Q trigger and per-SNR evaluation; adds faithful ports of four core
removers (retrain_delF, finetune_retain, recipe, nc_style), a recall/precision
sweep, per-row CSV flush + resume, gamma* estimation and two figures.

Data: real RadioML 2016.10a (Kaggle input
'nolasthitnotomorrow/radioml2016-deepsigcom', file RML2016.10a_dict.pkl),
all modulations present, full SNR range. Add Input -> Datasets -> Add.
"""
from nb_common import NB, NB_DIR

nb = NB()

nb.md(r"""
# FML - ECE/RF removal study: backdoor removal on RadioML AMC (3 seeds, ~1 h)

A paper-grade extension of the RF pilot. It ports the CIFAR recall/precision
backdoor-removal thesis to RF automatic modulation classification (AMC) and
sketches the P1 headline quantity **gamma\***: the detector recall needed to
reach defence success (ASR <= 5%) **as a function of SNR** - a curve with no
analogue in the vision literature.

### Dataset (no upload needed)
Real **RadioML 2016.10a** (DeepSig). In this Kaggle notebook:
1. Right panel -> **Add Input** -> **Datasets** -> search `radioml2016-deepsigcom` -> **Add**.
2. The code reads `RML2016.10a_dict.pkl` from `/kaggle/input` automatically.
All modulations present are used (11 for the standard file); the full SNR
range -20..18 dB is the domain-native axis.

### What it does
* **Model:** small 1-D CNN (Conv1d -> BN -> ReLU -> pool -> FC), kept tiny so the
  full grid fits the hour. Trains on one of the two T4s (DataParallel overhead
  is not worth it for ~50k params).
* **Attack:** BadNets-style I/Q trigger (fixed burst in the last 8 samples),
  dirty-label to BPSK, 5% poison.
* **Removal (4 faithful ports of the frozen CIFAR core):**
  `gold_retrain` (delete flagged, retrain from scratch), `finetune_retain`
  (delete flagged, fine-tune), `recipe` (centroid label-repair on the flagged
  set + trigger inoculation), `nc_style` (inoculation-only fine-tune).
* **Grid:** seeds [0,1,2]; recall x precision cells
  [(1.0,1.0), (0.75,1.0), (0.5,1.0), (0.75,0.7)].
* **Metrics:** clean accuracy + ASR overall and per SNR; **gamma\*** per
  method x SNR (min recall reaching ASR <= 5%, averaged over seeds).

### Resume (important for a ~1 h run)
Every finished row flushes immediately to `runs_rf_study.csv` (and per-SNR rows
to `rf_study_per_snr.csv`). Re-running Cell 5 in a live session skips finished
rows; a fully completed seed is skipped entirely. A full session restart begins
with an empty working directory.

### Settings
* Accelerator: **GPU T4 x 2**, Internet **On** (optional), Input: the RadioML dataset above.
* Runtime: **~45-70 min** on GPU T4 x2, calibrated from the pilot (~1 s/epoch per
  19.2k samples -> ~9 s/epoch at 176k train). Trim with `SEEDS`, `EPOCHS`,
  `FT_EPOCHS`, `CELLS` or `SNR_SET` if Kaggle threatens the session.

### Output
`FML_RF_Study.zip` -> `runs_rf_study.csv`, `rf_study_per_snr.csv`,
`gamma_star.csv`, `rf_study_asr_snr.png`, `rf_study_gamma_snr.png`,
`FML_RF_Study_meta.json`. Last cell shows a download link.

### Honesty
* A P1 **pilot**, not the full RF paper: one architecture, a synthetic I/Q
  trigger (not over-the-air), 4 cells (a coarse gamma* sketch, not the full
  recall x precision x SNR surface). State these limits in the paper.
* Differences < 1 pt are noise; gamma* is read off 3 recall levels, so treat it
  as a sketch. Weak ASR at low SNR is a result, not a bug - plot it.
""")

nb.code(r'''
# CELL 1 - knobs, environment, load RadioML 2016.10a (all mods, full SNR)
import os, sys, glob, json, time, pickle, platform, hashlib, zipfile, math
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

# ---- knobs ----
SEEDS      = [0, 1, 2]
RATE       = 0.05
EPOCHS     = 12          # base + gold_retrain
BS         = 256
LR         = 1e-3
FT_EPOCHS  = 4           # finetune_retain / recipe / nc_style
FT_LR      = 1e-4
SNR_SET    = list(range(-20, 20, 2))     # -20..18 dB, the full RadioML axis
CELLS      = [(1.0, 1.0), (0.75, 1.0), (0.5, 1.0), (0.75, 0.7)]
METHODS    = ['gold_retrain', 'finetune_retain', 'recipe', 'nc_style']
ASR_TARGET = 5.0         # defence-success threshold for gamma*
OUT        = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
dev        = 'cuda' if torch.cuda.is_available() else 'cpu'

print('python  ', sys.version.split()[0])
print('torch   ', torch.__version__, '| cuda', torch.cuda.is_available(),
      '| gpus', torch.cuda.device_count())
print('device  ', dev, '| out', OUT)
print('plan    ', 'seeds=%s epochs=%d ft=%d cells=%d methods=%d snrs=%d'
      % (SEEDS, EPOCHS, FT_EPOCHS, len(CELLS), len(METHODS), len(SNR_SET)))

def find_dataset():
    pats = ['/kaggle/input/**/RML2016.10a_dict.pkl',
            '/kaggle/input/**/*.pkl',
            './RML2016.10a_dict.pkl',
            './data/RML2016.10a_dict.pkl']
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(
        'RadioML not found. Use Add Input -> Datasets -> radioml2016-deepsigcom -> Add.')

def _mod(k):
    return k[0].decode() if isinstance(k[0], bytes) else str(k[0])

def load_radioml():
    path = find_dataset()
    print('loading', path)
    with open(path, 'rb') as f:
        Xd = pickle.load(f, encoding='latin1')
    # use every modulation present (robust to exact key spelling); 11 for RML2016.10a
    class_list = sorted({_mod(k) for k in Xd.keys()})
    mod2idx = {m: i for i, m in enumerate(class_list)}
    snrs_present = sorted({int(k[1]) for k in Xd.keys()})
    use_snr = [s for s in SNR_SET if s in snrs_present]
    assert use_snr, 'no requested SNR present; have %r' % snrs_present
    Xs, ys, ss = [], [], []
    for k, arr in Xd.items():
        m = _mod(k); s = int(k[1])
        if s not in use_snr:
            continue
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3 and a.shape[1] == 2:
            pass                      # (n, 2, 128) canonical
        elif a.ndim == 3 and a.shape[2] == 2:
            a = a.transpose(0, 2, 1)  # (n, 128, 2) -> (n, 2, 128)
        elif a.ndim == 2 and a.shape[1] == 256:
            a = a.reshape(-1, 2, 128)
        else:
            raise ValueError('unexpected I/Q shape %r' % (a.shape,))
        Xs.append(a)
        ys.append(np.full(len(a), mod2idx[m], np.int64))
        ss.append(np.full(len(a), s, np.int64))
    X = np.concatenate(Xs); y = np.concatenate(ys); snr = np.concatenate(ss)
    e = np.sqrt((X ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-8   # per-sample unit energy
    X = X / e
    return X, y, snr, class_list, use_snr

X_all, y_all, snr_all, CLASSES, SNR_SET = load_radioml()
NC = len(CLASSES)
TARGET = CLASSES.index('BPSK') if 'BPSK' in CLASSES else 0
# fixed train/test split (seed 0) so every seed is scored on the same test set;
# poison placement and model init vary per seed.
rng = np.random.RandomState(0)
perm = rng.permutation(len(X_all))
n_tr = int(0.8 * len(perm))
tr, te = perm[:n_tr], perm[n_tr:]
Xtr = torch.tensor(X_all[tr], device=dev); ytr = torch.tensor(y_all[tr], device=dev)
Xte = torch.tensor(X_all[te], device=dev); yte = torch.tensor(y_all[te], device=dev)
snte = snr_all[te]
print('classes (%d):' % NC, CLASSES)
print('train', tuple(Xtr.shape), 'test', tuple(Xte.shape), '| target =', CLASSES[TARGET])
print('SNR bins (test):', {int(s): int((snte == s).sum()) for s in SNR_SET})
''')

nb.code(r'''
# CELL 2 - model, trigger, poisoning, forget-set simulator, train/eval, 4 removers
class CNN1D(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(128, nc)

    def feat(self, x):
        return self.body(x).flatten(1)

    def forward(self, x):
        return self.fc(self.feat(x))

def add_trigger(x):
    x = x.clone()
    x[:, 0, 120:128] = 0.9
    x[:, 1, 120:128] = -0.9
    return x

def clone(base):
    m = CNN1D(NC).to(dev)
    m.load_state_dict(base.state_dict())
    return m

def poison_dataset(X, y, rate, seed, target=TARGET):
    g = np.random.RandomState(seed)
    cand = np.where(y.cpu().numpy() != target)[0]
    n_poison = int(rate * len(y))
    sel = g.choice(cand, size=min(n_poison, len(cand)), replace=False)
    Xp, yp = X.clone(), y.clone()
    sel_t = torch.from_numpy(sel).to(X.device)
    Xp[sel_t] = add_trigger(X[sel_t])
    yp[sel_t] = target
    ip = torch.zeros(len(y), dtype=torch.bool, device=X.device)
    ip[sel_t] = True
    return Xp, yp, ip

def make_forget_set(is_poison, y, recall, precision, seed, target=TARGET):
    ip = is_poison.cpu().numpy() if isinstance(is_poison, torch.Tensor) else np.asarray(is_poison)
    yn = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    p_idx = np.where(ip)[0]
    n_p = int(round(recall * len(p_idx)))
    n_f = int(round(n_p / max(precision, 1e-9)))
    n_c = max(0, n_f - n_p)
    clean_pool = np.where((~ip) & (yn != target))[0]
    rng = np.random.RandomState(seed * 1009 + 17)
    Fp = rng.choice(p_idx, size=min(n_p, len(p_idx)), replace=False)
    Fc = rng.choice(clean_pool, size=min(n_c, len(clean_pool)), replace=False)
    mask = np.zeros(len(yn), bool); mask[Fp] = True; mask[Fc] = True
    stats = dict(recall=round(len(Fp) / max(len(p_idx), 1), 4),
                 precision=round(len(Fp) / mask.sum(), 4) if mask.sum() else 1.0,
                 n_forget=int(mask.sum()))
    return mask, stats

def train_model(X, y, epochs, lr, model=None, tag='', seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    if model is None:
        model = CNN1D(NC).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(y), device=dev)
        tot = 0.0; nb = 0
        for k in tqdm(range(0, len(y), BS), desc='%s ep%d' % (tag, ep + 1), unit='batch', leave=False):
            idx = perm[k:k + BS]
            if len(idx) < 4:
                continue
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print('  [%s] ep %d/%d loss=%.4f (%.0fs)' % (tag, ep + 1, epochs, tot / max(1, nb), time.time() - t0))
    model.eval()
    return model

@torch.no_grad()
def extract_features(model, X, bs=1024):
    model.eval()
    return torch.cat([model.feat(X[k:k + bs]) for k in range(0, len(X), bs)])

@torch.no_grad()
def predict(model, X, bs=1024):
    model.eval()
    return torch.cat([model(X[k:k + bs]).argmax(1) for k in range(0, len(X), bs)])

@torch.no_grad()
def evaluate(model, X, y, snr):
    preds = predict(model, X).cpu().numpy()
    yn = y.cpu().numpy()
    acc = float((preds == yn).mean() * 100)
    nt = yn != TARGET
    xt = add_trigger(X[torch.from_numpy(np.where(nt)[0]).to(dev)])
    asr = float((predict(model, xt).cpu().numpy() == TARGET).mean() * 100)
    per = {}
    for s in SNR_SET:
        m = snr == s
        if m.sum() == 0:
            continue
        a = float((preds[m] == yn[m]).mean() * 100)
        ntm = m & nt
        if ntm.sum() > 0:
            xtm = add_trigger(X[torch.from_numpy(np.where(ntm)[0]).to(dev)])
            ar = float((predict(model, xtm).cpu().numpy() == TARGET).mean() * 100)
        else:
            ar = float('nan')
        per[int(s)] = dict(clean_acc=round(a, 2), asr=round(ar, 2))
    return round(acc, 2), round(asr, 2), per

# ---- four faithful ports of the frozen CIFAR core removers ----
def remove_gold(base, X, y, Fmask, seed):
    Ft = torch.from_numpy(Fmask).to(dev)
    return train_model(X[~Ft], y[~Ft], EPOCHS, LR, tag='gold_s%d' % seed, seed=seed)

def remove_finetune(base, X, y, Fmask, seed):
    Ft = torch.from_numpy(Fmask).to(dev)
    return train_model(X[~Ft], y[~Ft], FT_EPOCHS, FT_LR, model=clone(base),
                       tag='ft_s%d' % seed, seed=seed)

def _innocence(X, y, retain, seed, n=5000):
    yn = y.cpu().numpy()
    pool = retain[yn[retain] != TARGET]
    if len(pool) > n:
        pool = np.random.RandomState(seed).choice(pool, n, replace=False)
    it = torch.from_numpy(pool).to(dev)
    return it, add_trigger(X[it]), y[it]

def remove_recipe(base, X, y, Fmask, seed):
    # centroid label-repair on the flagged set + trigger inoculation, then fine-tune
    torch.manual_seed(seed); np.random.seed(seed)
    retain = np.where(~Fmask)[0]; flag = np.where(Fmask)[0]
    rt = torch.from_numpy(retain).to(dev); ft = torch.from_numpy(flag).to(dev)
    with torch.no_grad():
        fr = extract_features(base, X[rt]); yr = y[rt]
        centroids = torch.stack([
            fr[yr == c].mean(0) if (yr == c).any() else torch.zeros(fr.shape[1], device=dev)
            for c in range(NC)])
        pseudo_flag = torch.cdist(extract_features(base, X[ft]), centroids).argmin(1)
    pseudo = y.clone(); pseudo[ft] = pseudo_flag
    it, X_in, y_in = _innocence(X, y, retain, seed)
    m = clone(base)
    opt = torch.optim.Adam(m.parameters(), lr=FT_LR, weight_decay=1e-4)
    m.train()
    n_batches = math.ceil(len(y) / BS)
    for ep in range(FT_EPOCHS):
        loss_val = 0.0
        for b in tqdm(range(n_batches), desc='recipe_s%d ep%d' % (seed, ep + 1), unit='batch', leave=False):
            nf = max(1, int(BS * 0.2)); nr = int(BS * 0.6); ni = BS - nr - nf
            bi = rt[torch.randint(0, len(rt), (nr,), device=dev)]
            fi = ft[torch.randint(0, len(ft), (nf,), device=dev)]
            ii = torch.randint(0, len(it), (ni,), device=dev)
            xb = torch.cat([X[bi], X[fi], X_in[ii]])
            yb = torch.cat([y[bi], pseudo[fi], y_in[ii]])
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_val = loss.item()
        print('  [recipe_s%d] ep %d/%d loss=%.4f' % (seed, ep + 1, FT_EPOCHS, loss_val))
    m.eval()
    return m

def remove_nc(base, X, y, Fmask, seed):
    # inoculation-only: fine-tune on retain + trigger-stamped clean samples (no repair)
    torch.manual_seed(seed); np.random.seed(seed)
    retain = np.where(~Fmask)[0]
    rt = torch.from_numpy(retain).to(dev)
    it, X_in, y_in = _innocence(X, y, retain, seed)
    m = clone(base)
    opt = torch.optim.Adam(m.parameters(), lr=FT_LR, weight_decay=1e-4)
    m.train()
    n_batches = math.ceil(len(retain) / BS)
    for ep in range(FT_EPOCHS):
        loss_val = 0.0
        for b in tqdm(range(n_batches), desc='nc_s%d ep%d' % (seed, ep + 1), unit='batch', leave=False):
            nr = int(BS * 0.75); ni = BS - nr
            bi = rt[torch.randint(0, len(rt), (nr,), device=dev)]
            ii = torch.randint(0, len(it), (ni,), device=dev)
            xb = torch.cat([X[bi], X_in[ii]])
            yb = torch.cat([y[bi], y_in[ii]])
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_val = loss.item()
        print('  [nc_s%d] ep %d/%d loss=%.4f' % (seed, ep + 1, FT_EPOCHS, loss_val))
    m.eval()
    return m

def apply_remove(method, base, X, y, Fmask, seed):
    if method == 'gold_retrain':
        return remove_gold(base, X, y, Fmask, seed)
    if method == 'finetune_retain':
        return remove_finetune(base, X, y, Fmask, seed)
    if method == 'recipe':
        return remove_recipe(base, X, y, Fmask, seed)
    if method == 'nc_style':
        return remove_nc(base, X, y, Fmask, seed)
    raise ValueError(method)

def cleanup():
    if dev == 'cuda':
        torch.cuda.empty_cache()

print('RF study functions ready: CNN1D, add_trigger, poison_dataset, make_forget_set,')
print('  train_model, evaluate, removers =', METHODS)
''')

nb.code(r'''
# CELL 3 - runner with per-row CSV flush + resume
CSV = 'runs_rf_study.csv'
PERCSV = 'rf_study_per_snr.csv'
COLS = ['exp', 'dataset', 'seed', 'method', 'recall', 'precision', 'n_forget',
        'clean_acc', 'asr', 'time_sec', 'notes']
PERCOLS = ['seed', 'method', 'recall', 'precision', 'snr', 'clean_acc', 'asr']

class Logger:
    def __init__(self, path, cols, keycols):
        self.path = path; self.cols = cols; self.keycols = keycols
        self.df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)
        self.keys = set()
        if len(self.df):
            for row in self.df[keycols].itertuples(index=False, name=None):
                self.keys.add(tuple(row))

    def has(self, **key):
        return tuple(key[k] for k in self.keycols) in self.keys

    def log(self, **row):
        k = tuple(row[c] for c in self.keycols)
        if k in self.keys:
            return
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.keys.add(k)
        self.df.to_csv(self.path, index=False)

def log_per(PLOG, seed, method, recall, precision, per):
    for s, d in per.items():
        PLOG.log(seed=seed, method=method, recall=recall, precision=precision,
                 snr=int(s), clean_acc=d['clean_acc'], asr=d['asr'])

def run_seed(seed, LOG, PLOG):
    t_seed = time.time()
    print('\n===== seed %d =====' % seed)
    Xp, yp, ip = poison_dataset(Xtr, ytr, RATE, seed)
    n_poison = int(ip.sum())
    t0 = time.time()
    base = train_model(Xp, yp, EPOCHS, LR, tag='base_s%d' % seed, seed=seed)
    acc0, asr0, per0 = evaluate(base, Xte, yte, snte)
    LOG.log(exp='RF', dataset='RadioML2016.10a', seed=seed, method='base',
            recall=1.0, precision=1.0, n_forget=n_poison, clean_acc=acc0, asr=asr0,
            time_sec=round(time.time() - t0, 1),
            notes='poisoned base; n_poison=%d; target=%s; IQ patch last 8 samples' % (n_poison, CLASSES[TARGET]))
    log_per(PLOG, seed, 'base', 1.0, 1.0, per0)
    print('  [base_s%d] acc=%.2f asr=%.2f n_poison=%d' % (seed, acc0, asr0, n_poison))

    for (r, p) in tqdm(CELLS, desc='cells_s%d' % seed, unit='cell', leave=False):
        Fmask, st = make_forget_set(ip, yp, r, p, seed)
        for meth in METHODS:
            if LOG.has(seed=seed, method=meth, recall=st['recall'], precision=st['precision']):
                continue
            t0 = time.time()
            m = apply_remove(meth, base, Xp, yp, Fmask, seed)
            acc, asr, per = evaluate(m, Xte, yte, snte)
            dt = time.time() - t0
            LOG.log(exp='RF', dataset='RadioML2016.10a', seed=seed, method=meth,
                    recall=st['recall'], precision=st['precision'], n_forget=st['n_forget'],
                    clean_acc=acc, asr=asr, time_sec=round(dt, 1),
                    notes='F(r,p) port of core %s; target=%s' % (meth, CLASSES[TARGET]))
            log_per(PLOG, seed, meth, st['recall'], st['precision'], per)
            print('  [log] s%d %-16s r=%.2f p=%.2f acc=%.2f asr=%.2f (%.0fs)'
                  % (seed, meth, st['recall'], st['precision'], acc, asr, dt))
            del m; cleanup()
    del base, Xp; cleanup()
    print('seed %d done in %.1f min' % (seed, (time.time() - t_seed) / 60))

def run_all():
    LOG = Logger(CSV, COLS, ['seed', 'method', 'recall', 'precision'])
    PLOG = Logger(PERCSV, PERCOLS, ['seed', 'method', 'recall', 'precision', 'snr'])
    t_all = time.time()
    expected_per_seed = 1 + len(CELLS) * len(METHODS)
    for seed in tqdm(SEEDS, desc='seeds', unit='seed'):
        existing = int((LOG.df.seed == seed).sum()) if len(LOG.df) else 0
        if existing >= expected_per_seed:
            print('seed %d already complete (%d rows) - skipping' % (seed, existing))
            continue
        run_seed(seed, LOG, PLOG)
    print('\nall done in %.1f min' % ((time.time() - t_all) / 60))
''')

nb.code(r'''
# CELL 4 - the long cell (~45-70 min). Re-running resumes from the CSV.
run_all()
''')

nb.code(r'''
# CELL 5 - gamma* estimation, figures, audit, package, download link
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import FileLink, display

df = pd.read_csv(CSV)
ps = pd.read_csv(PERCSV)

# ---- gamma*: min recall reaching ASR<=target, per method x SNR (precision=1 cells) ----
prec1 = ps[ps.precision == 1.0].copy()
prec1['recall_r'] = prec1.recall.round(2)
g_rows = []
for meth in METHODS:
    for s in sorted(ps.snr.unique()):
        sub = prec1[(prec1.method == meth) & (prec1.snr == s)]
        if not len(sub):
            continue
        avg = sub.groupby('recall_r').asr.mean()
        good = avg[avg <= ASR_TARGET]
        gamma = float(good.index.min()) if len(good) else float('nan')
        g_rows.append(dict(method=meth, snr=int(s), gamma_star=gamma,
                           asr_oracle=round(float(avg.get(1.0, float('nan'))), 2),
                           min_asr=round(float(avg.min()), 2)))
gs = pd.DataFrame(g_rows)
gs.to_csv('gamma_star.csv', index=False)
print('--- gamma_star.csv (min recall for ASR<=%.1f; NaN = not removable even at oracle) ---' % ASR_TARGET)
print(gs.to_string(index=False))

# ---- figure 1: ASR vs SNR at the oracle cell, seed-averaged ----
orcl = ps[(ps.precision == 1.0) & (ps.recall.round(2) == 1.0)]
fig, ax = plt.subplots(figsize=(8, 4.8))
styles = {'base': ('o-', 'poisoned base'),
          'gold_retrain': ('s-', 'gold retrain (oracle)'),
          'finetune_retain': ('^-', 'fine-tune retain (oracle)'),
          'recipe': ('d-', 'recipe (oracle)'),
          'nc_style': ('v-', 'NC-style inoculation (oracle)')}
for meth, (fmt, label) in styles.items():
    sub = orcl[orcl.method == meth].groupby('snr').asr.mean().sort_index()
    if len(sub):
        ax.plot(sub.index, sub.values, fmt, label=label)
ax.axhline(ASR_TARGET, ls='--', c='grey', lw=1, label='defence success (ASR<=%.0f%%)' % ASR_TARGET)
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Attack success rate (%)')
ax.set_title('RF AMC: ASR vs SNR (oracle removal, mean over seeds)')
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig('rf_study_asr_snr.png', dpi=150); plt.close(fig)

# ---- figure 2: gamma* vs SNR per method (the P1 headline) ----
fig, ax = plt.subplots(figsize=(8, 4.8))
for meth, (fmt, label) in styles.items():
    if meth == 'base':
        continue
    sub = gs[gs.method == meth].sort_values('snr')
    if len(sub):
        ax.plot(sub.snr, sub.gamma_star, fmt, label=label)
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('gamma* = min detector recall for ASR<=%.0f%%' % ASR_TARGET)
ax.set_title('RF AMC: recall tolerance gamma* vs SNR (coarse sketch)')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig('rf_study_gamma_snr.png', dpi=150); plt.close(fig)
print('saved rf_study_asr_snr.png, rf_study_gamma_snr.png')

# ---- audit ----
print('\n--- %s ---' % CSV)
print(df.to_string(index=False))
key_cols = ['seed', 'method', 'recall', 'precision']
dup = int(df.duplicated(subset=key_cols).sum())
bad = int((df.clean_acc.isna() | df.asr.isna()).sum())
expected = len(SEEDS) * (1 + len(CELLS) * len(METHODS))
ok = (len(df) == expected) and dup == 0 and bad == 0
print('\nrows %d | expected %d | duplicate keys %d | missing acc/asr %d -> %s'
      % (len(df), expected, dup, bad, 'PASS' if ok else 'CHECK'))

meta = dict(purpose='ECE/RF removal study: recall/precision backdoor removal on RadioML AMC + gamma*',
            dataset='RadioML2016.10a (Kaggle nolasthitnotomorrow/radioml2016-deepsigcom)',
            classes=CLASSES, n_classes=NC, snr_set=SNR_SET, target=CLASSES[TARGET],
            knobs=dict(seeds=SEEDS, rate=RATE, epochs=EPOCHS, bs=BS, lr=LR,
                       ft_epochs=FT_EPOCHS, ft_lr=FT_LR, cells=CELLS, methods=METHODS,
                       asr_target=ASR_TARGET),
            rows_expected=expected, rows_actual=int(len(df)), audit_pass=bool(ok),
            generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            host=platform.platform())
json.dump(meta, open('FML_RF_Study_meta.json', 'w'), indent=2)

with zipfile.ZipFile('FML_RF_Study.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in [CSV, PERCSV, 'gamma_star.csv', 'rf_study_asr_snr.png',
              'rf_study_gamma_snr.png', 'FML_RF_Study_meta.json']:
        if os.path.exists(f):
            z.write(f)

print('\n============================================================')
print('DOWNLOAD: /kaggle/working/FML_RF_Study.zip')
print('Click below, or Output tab -> FML_RF_Study.zip -> Download.')
print('============================================================')
display(FileLink('FML_RF_Study.zip'))
''')

nb.write(NB_DIR / 'FML_RF_Removal_Study.ipynb')