"""
Build FML_Sai_Kaggle_Run.ipynb from the frozen core_kaggle.py + person_sai.py.

The notebook writes both files with %%writefile, then exec()s them into the
notebook namespace with the driver's __main__ block disabled. That reproduces
the documented "Cell 1 = core, Cell 2 = driver" layout without the auto-run
that a bare paste triggers (Jupyter sets __name__ == '__main__').

Usage:  python make_sai_notebook.py [out.ipynb]
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = r'E:\FML\handouts\notebooks'
CORE = os.path.join(HERE, 'core_kaggle.py')
DRIVER = os.path.join(HERE, 'person_sai.py')

CORE_LF_SHA = 'ad7a363926a5'
DRIVER_LF_SHA = '45295331c333'


def read_lf(path):
    b = open(path, 'rb').read().replace(b'\r\n', b'\n')
    t = b.decode('utf-8')
    if not t.endswith('\n'):
        t += '\n'
    return t


def sha12(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


core_txt = read_lf(CORE)
drv_txt = read_lf(DRIVER)
assert sha12(core_txt) == CORE_LF_SHA, sha12(core_txt)
assert sha12(drv_txt) == DRIVER_LF_SHA, sha12(drv_txt)
assert drv_txt.count("if __name__ == '__main__':") == 1


def lines(s):
    return s.splitlines(keepends=True)


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines(s)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(s)}


# ------------------------------------------------------------------ cells
CELLS = []

CELLS.append(md(r"""# FML · Sai — Kaggle run notebook (seed 0)

**What this is.** A self-contained notebook for the Sai work package: **E1** (main detector grid), **E9** (amplification at scale, 3 seeds), **E3** (Calibrate-then-Remove), **E6** (realistic detector column), **E7b** (inoculation with a synthesised trigger). Target: 96–97 rows in `runs_sai.csv`.

**Requirements**
- Kaggle notebook, **Accelerator = GPU T4 x2**, **Internet = On** (or a CIFAR-10 dataset attached under Input).
- One interactive session with the tab open. Expected wall-clock **5.5–6.5 hours**; Kaggle stops a GPU session at about 9 hours.

**How to run, in order**

| Cell | What it does | Time |
|---|---|---|
| 1 | environment check (GPU, disk, working dir) | 5 s |
| 2 | resume: restores `runs_sai.csv` if an input dataset has one | 5 s |
| 3 | writes `core_kaggle.py` (frozen, do not edit) | 1 s |
| 4 | writes `person_sai.py` (frozen, do not edit) | 1 s |
| 5 | verifies both files against the frozen hashes and compiles them | 2 s |
| 6 | loads the code, checks CIFAR shapes, one forward/backward smoke | 1–2 min |
| 7 | **runs all five blocks** — the long cell | 5.5–6.5 h |
| 8 | audit, gate summary, `FML_Sai_results.zip` | 20 s |

The manual numbers its cells differently because it has no separate environment or preflight cells: manual Cell 1 = this Cell 3, manual Cell 2 = this Cell 4, and manual Cell 3 = this Cells 5–7. When you work inside this file, follow the notebook's numbering.

**Resume.** Every measured row is flushed to `/kaggle/working/runs_sai.csv` immediately. If the session dies, download that CSV, upload it to a private dataset named `fml-runs-sai`, attach the dataset, and run Cells 1 → 7 again. The resume key skips every completed row, so nothing is measured twice. One exception: **E6 re-runs whenever you re-run the block**, because its frozen driver logs `detector='none'`; the Logger still refuses to duplicate the rows, but the block costs about 30 minutes again.

**Do not edit** `core_kaggle.py` or `person_sai.py` inside this notebook. Cell 5 proves the text you ran is the text that was frozen.
"""))

CELLS.append(code(r"""# CELL 1 - environment check (read-only, ~5 s)
import os, sys, platform, shutil

print('python  :', sys.version.split()[0], '|', platform.platform())
try:
    import torch, torchvision
    print('torch   :', torch.__version__, '| torchvision', torchvision.__version__)
    print('cuda    :', torch.cuda.is_available(), '| build', torch.version.cuda)
    n = torch.cuda.device_count()
    print('gpus    :', n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print('          gpu %d: %s | %.1f GB | sm_%d%d'
              % (i, p.name, p.total_memory / 1e9, p.major, p.minor))
    if n == 0:
        print('')
        print('STOP: no GPU visible. Right panel -> Session options -> Accelerator -> GPU T4 x2,')
        print('      then restart the kernel and run again from CELL 1.')
    elif n == 1:
        print('')
        print('WARNING: one GPU. The frozen protocol is 2x T4; timings will roughly double.')
except Exception as e:
    print('TORCH IMPORT FAILED:', repr(e))

w = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
du = shutil.disk_usage(w)
print('workdir :', os.path.abspath(w), '| free %.1f GB' % (du.free / 1e9))
print('files   :', sorted(os.listdir(w))[:30])
"""))

CELLS.append(code(r"""# CELL 2 - resume from an earlier session. Safe to run every time, ~5 s.
import os, glob, shutil
import pandas as pd

W = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
CSV = os.path.join(W, 'runs_sai.csv')


def _rows(p):
    try:
        return len(pd.read_csv(p))
    except Exception:
        return -1


srcs = sorted(glob.glob('/kaggle/input/**/runs_sai.csv', recursive=True))
if not srcs:
    print('no runs_sai.csv under /kaggle/input - starting fresh')
for s in srcs:
    if os.path.abspath(s) == os.path.abspath(CSV):
        continue
    have = _rows(CSV) if os.path.exists(CSV) else -1
    new = _rows(s)
    if new > have:
        shutil.copy(s, CSV)
        print('restored %s (%d rows) -> %s' % (s, new, CSV))
    else:
        print('kept local CSV (%d rows); %s has %d' % (have, s, new))

for pat in ('ckpt_*.pt', 'synth_*.pt'):
    for s in glob.glob('/kaggle/input/**/' + pat, recursive=True):
        d = os.path.join(W, os.path.basename(s))
        if not os.path.exists(d):
            shutil.copy(s, d)
            print('restored', os.path.basename(s))

print('runs_sai.csv:', ('present, %d rows' % _rows(CSV)) if os.path.exists(CSV) else 'not present')
"""))

CELLS.append(code('%%writefile core_kaggle.py\n' + core_txt))
CELLS.append(code('%%writefile person_sai.py\n' + drv_txt))

CELLS.append(code(r"""# CELL 5 - integrity + syntax check (read-only, ~2 s)
import hashlib, os

# sha256[:12] of the LF-normalised text, which is exactly what %%writefile produces on Kaggle.
# The Windows copies of the same files hash to 67b5474e75f2 (core) because they use CRLF.
EXPECT = {'core_kaggle.py': 'ad7a363926a5', 'person_sai.py': '45295331c333'}
ok = True
for f, exp in EXPECT.items():
    assert os.path.exists(f), f + ' missing - run the cell that writes it'
    b = open(f, 'rb').read()
    got = hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest()[:12]
    hit = (got == exp)
    ok = ok and hit
    print('%-18s %6d bytes  sha256[:12]=%s  expected=%s  %s'
          % (f, len(b), got, exp, 'OK' if hit else 'MISMATCH'))
    compile(b.decode('utf-8'), f, 'exec')
print('syntax: OK')
if not ok:
    raise RuntimeError('hash mismatch: this is not the frozen text. Stop and report; do not edit the file to match.')
"""))

CELLS.append(code(r"""# CELL 6 - load the frozen code into this notebook, then check data and GPUs (~1-2 min first time)
import os, torch, numpy as np

G = globals()
exec(compile(open('core_kaggle.py').read(), 'core_kaggle.py', 'exec'), G)

_src = open('person_sai.py').read()
_guard = "if __name__ == '__main__':"
assert _src.count(_guard) == 1, 'driver main-guard not found exactly once'
exec(compile(_src.replace(_guard, 'if False:  # notebook: blocks are called explicitly in CELL 7'),
             'person_sai.py', 'exec'), G)
print('core + driver loaded; driver main block disabled')

setup('sai', seed=0)
load_cifar()
assert dev == 'cuda', 'No GPU. Session options -> Accelerator -> GPU T4 x2, restart, rerun from CELL 1.'
assert tuple(Xtr.shape) == (50000, 3, 32, 32), Xtr.shape
assert tuple(Xte.shape) == (10000, 3, 32, 32), Xte.shape
print('data    :', tuple(Xtr.shape), tuple(Xte.shape),
      '| dev', dev, '| gpus', NGPU, '| bs', BS, '| epochs', EPOCHS)

m = CIFARModel('resnet18').to(dev)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
loss = F.cross_entropy(m(augment_batch(Xtr[:64])), ytr[:64])
loss.backward()
opt.step()
print('smoke   : forward/backward OK, loss %.4f' % float(loss))
del m, opt
cleanup()
print('READY')
"""))

CELLS.append(code(r"""# CELL 7 - run all five blocks. Long: 5.5-6.5 h. Safe to re-run; completed rows are skipped.
import os, time, zipfile, traceback

BLOCKS = ['E1', 'E9', 'E3', 'E6', 'E7b']
MAX_HOURS = 8.0          # headroom under Kaggle's ~9 h GPU session cap
CALLS = {
    'E1':  lambda: run_e1(LOG, 0),
    'E9':  lambda: run_e9(LOG, seeds=(0, 1, 2)),
    'E3':  lambda: run_e3(LOG, 0),
    'E6':  lambda: run_e6(LOG, 0),
    'E7b': lambda: run_e7b(LOG, 0),
}

if 'boot' not in globals():
    raise RuntimeError('run CELL 6 first: the frozen core is not loaded in this kernel')


def snapshot():
    csv = os.path.join(OUT, 'runs_sai.csv')
    if not os.path.exists(csv):
        return
    z = os.path.join(OUT, 'FML_Sai_results_partial.zip')
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv, 'runs_sai.csv')
        for f in ('core_kaggle.py', 'person_sai.py'):
            if os.path.exists(f):
                zf.write(f, f)
    print('  snapshot -> %s (%.0f KB)' % (z, os.path.getsize(z) / 1024))


LOG = boot()
print('rows already logged:', len(LOG.rows))

RUN_STATUS = {}
t_all = time.time()
for blk in BLOCKS:
    hrs = (time.time() - t_all) / 3600
    if hrs > MAX_HOURS:
        RUN_STATUS[blk] = 'SKIPPED (time budget)'
        print('\n===== %s skipped: %.1f h already spent =====' % (blk, hrs))
        continue
    print('\n===== %s start (%.2f h elapsed) =====' % (blk, hrs), flush=True)
    t0 = time.time()
    try:
        CALLS[blk]()
        RUN_STATUS[blk] = 'ok (%.1f min)' % ((time.time() - t0) / 60)
    except Exception as e:
        traceback.print_exc()
        RUN_STATUS[blk] = 'FAILED %s: %s' % (type(e).__name__, e)
    finally:
        try:
            cleanup()
        except Exception:
            pass
    print('===== %s %s =====' % (blk, RUN_STATUS[blk]), flush=True)
    print('  csv rows now:', len(LOG.rows))
    snapshot()

print('\nblock status:')
for k in BLOCKS:
    print('  %-4s %s' % (k, RUN_STATUS.get(k, 'not reached')))
print('total %.2f h' % ((time.time() - t_all) / 3600))
"""))

CELLS.append(code(r"""# CELL 8 - audit + gate summary + zip. Never edits the CSV.
import os, json, time, platform, zipfile, hashlib
import numpy as np, pandas as pd, torch

W = globals().get('OUT') or ('/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
CSV = os.path.join(W, 'runs_sai.csv')
assert os.path.exists(CSV), 'runs_sai.csv not found - run CELL 7 first'
df = pd.read_csv(CSV)

KEY = ['exp', 'arch', 'attack', 'rate', 'seed', 'recall', 'precision',
       'n_forget', 'method', 'detector']
EXPECT = {'E1': (53, 53), 'E9': (15, 15), 'E3': (21, 22), 'E6': (6, 6), 'E7b': (1, 1)}
NO_ACC = {'poisoned', 'mia', 'relearn', 'label_acc', 'car_proxy'}

lines = []


def say(s=''):
    print(s)
    lines.append(str(s))


def f2(v):
    try:
        return '%.2f' % float(v)
    except Exception:
        return 'NA'


def f4(v):
    try:
        return '%.4f' % float(v)
    except Exception:
        return 'NA'


def num(row, k):
    if row is None:
        return None
    try:
        v = float(row[k])
        return v if np.isfinite(v) else None
    except Exception:
        return None


def parse_notes(s):
    out = {}
    for tok in str(s).replace(';', ' ').split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            try:
                out[k] = float(v)
            except Exception:
                out[k] = v
    return out


def one(exp, method, r=None, p=None, seed=None, detector=None):
    d = df[df.exp == exp]
    if method is not None:
        d = d[d.method == method]
    if detector is not None:
        d = d[d.detector == detector]
    if seed is not None:
        d = d[d.seed == seed]
    if r is not None:
        d = d[np.isclose(pd.to_numeric(d.recall, errors='coerce'), r, atol=2e-3)]
    if p is not None:
        d = d[np.isclose(pd.to_numeric(d.precision, errors='coerce'), p, atol=2e-3)]
    return d.iloc[0] if len(d) else None


say('FML / Sai (seed 0) - audit')
say('=' * 64)
say('rows: %d   columns: %d' % (len(df), len(df.columns)))
say('')
say('rows per experiment (measured / expected):')
for e in ['E1', 'E9', 'E3', 'E6', 'E7b']:
    n = int((df.exp == e).sum())
    lo, hi = EXPECT[e]
    flag = 'OK' if lo <= n <= hi else ('LOW' if n < lo else 'HIGH')
    say('  %-4s %4d / %d-%d   %s' % (e, n, lo, hi, flag))
say('')

problems = []
dup = int(df.duplicated(KEY).sum())
if dup:
    problems.append('%d duplicate resume keys' % dup)
for c in ['rate', 'recall', 'precision']:
    v = pd.to_numeric(df[c], errors='coerce')
    bad = int(((v < 0) | (v > 1)).sum())
    if bad:
        problems.append('%d rows with %s outside [0,1]' % (bad, c))
rem = df[~df.method.isin(NO_ACC)]
for c in ['clean_acc', 'asr']:
    v = pd.to_numeric(rem[c], errors='coerce')
    bad = int((v.isna() | (v < 0) | (v > 100)).sum())
    if bad:
        problems.append('%d removal rows with %s missing/out of [0,100]' % (bad, c))
say('duplicate resume keys: %d  (must be 0)' % dup)
say('removal rows: %d' % len(rem))
say('AUDIT: ' + ('PASS' if not problems else 'FAIL - ' + '; '.join(problems)))
say('')

send = []
GATES = {}

say('GATE 1 - E1 at recall 0.75, precision 1.0')
rec = one('E1', 'recipe', 0.75, 1.0)
rt = one('E1', 'retrain_delF', 0.75, 1.0)
if rec is not None and rt is not None:
    say('  recipe        ASR %s  acc %s  (n_forget %d)'
        % (f2(num(rec, 'asr')), f2(num(rec, 'clean_acc')), int(float(rec.n_forget))))
    say('  retrain_delF  ASR %s  acc %s' % (f2(num(rt, 'asr')), f2(num(rt, 'clean_acc'))))
    gate1 = (num(rec, 'asr') is not None and num(rt, 'asr') is not None
             and num(rec, 'asr') < 20 and num(rt, 'asr') > 90)
    GATES['E1_075'] = dict(recipe_asr=num(rec, 'asr'), recipe_acc=num(rec, 'clean_acc'),
                           retrain_asr=num(rt, 'asr'), retrain_acc=num(rt, 'clean_acc'),
                           pass_=bool(gate1))
    say('  GATE: ' + ('PASS - recipe degrades gracefully where retraining collapses'
                      if gate1 else 'FAIL - pattern not visible; report the numbers, do not tune'))
    send.append('E1 (0.75,1.0): recipe ASR %s acc %s | retrain_delF ASR %s acc %s'
                % (f2(num(rec, 'asr')), f2(num(rec, 'clean_acc')),
                   f2(num(rt, 'asr')), f2(num(rt, 'clean_acc'))))
else:
    say('  MISSING rows - rerun E1')
say('')

orc = one('E1', 'retrain_delF', 1.0, 1.0)
poi = one('E1', 'poisoned', 1.0, 1.0, detector='ss')
say('GATE 1b - E1 oracle (recall 1.0, precision 1.0)')
if orc is not None:
    say('  retrain_delF  ASR %s  acc %s' % (f2(num(orc, 'asr')), f2(num(orc, 'clean_acc'))))
    say('  GATE: ' + ('PASS' if (num(orc, 'asr') or 0) <= 5 else 'CHECK - oracle ASR above 5'))
    send.append('E1 (1.0,1.0) oracle: retrain_delF ASR %s acc %s'
                % (f2(num(orc, 'asr')), f2(num(orc, 'clean_acc'))))
if poi is not None:
    say('  poisoned base: AUC %s  tpr@fpr10 %s  flags %s / %s true'
        % (f4(num(poi, 'auc')), f4(num(poi, 'tpr_at_fpr10')),
           int(float(poi.n_flags)) if str(poi.n_flags) != '' else 'NA',
           int(float(poi.n_true_in_flags)) if str(poi.n_true_in_flags) != '' else 'NA'))
say('')

say('E1 grid - ASR by (recall, precision) and method')
piv = df[(df.exp == 'E1') & df.method.isin(['retrain_delF', 'finetune_retain',
                                            'negrad_plus', 'scrub', 'recipe'])]
if len(piv):
    t = piv.pivot_table(index=['recall', 'precision'], columns='method',
                        values='asr', aggfunc='first')
    say(t.round(2).to_string())
say('')

say('GATE 2 - E9 amplification (3 seeds, 10% poison)')
d9 = df[df.exp == 'E9']
e9_pass = []
for seed in [0, 1, 2]:
    d = d9[d9.seed == seed]
    base = d[d.method == 'base']
    if len(base) == 0:
        say('  seed %d: base row missing' % seed)
        continue
    ba = num(base.iloc[0], 'auc')
    bacc = num(base.iloc[0], 'clean_acc')
    say('  seed %d base AUC %s  acc %s' % (seed, f4(ba), f2(bacc)))
    best, ctrl_best = None, None
    for _, r in d[d.method.str.startswith('partial_repair')].iterrows():
        da = ((num(r, 'auc') - ba) * 100) if (ba is not None and num(r, 'auc') is not None) else None
        loss = (bacc - num(r, 'clean_acc')) if (bacc is not None and num(r, 'clean_acc') is not None) else None
        say('    %-30s AUC %s  dAUC %s pt  acc %s  dacc_loss %s'
            % (r.method, f4(num(r, 'auc')), ('%+.2f' % da) if da is not None else 'NA',
               f2(num(r, 'clean_acc')), f2(loss)))
        if da is not None and (best is None or da > best[0]):
            best = (da, loss if loss is not None else 99.0, r.method)
    for _, r in d[d.method.str.startswith('ctrl_clean')].iterrows():
        da = ((num(r, 'auc') - ba) * 100) if (ba is not None and num(r, 'auc') is not None) else None
        say('    %-30s AUC %s  dAUC %s pt  acc %s'
            % (r.method, f4(num(r, 'auc')), ('%+.2f' % da) if da is not None else 'NA',
               f2(num(r, 'clean_acc'))))
        if da is not None and (ctrl_best is None or da > ctrl_best):
            ctrl_best = da
    if best is not None and ctrl_best is not None:
        passed = best[0] >= 10 and best[1] <= 2 and best[0] > ctrl_best
        e9_pass.append(passed)
        say('  SEED %d GATE: %s  (best dAUC %+.2f pt, acc loss %.2f, best ctrl dAUC %+.2f pt)'
            % (seed, 'PASS' if passed else 'FAIL', best[0], best[1], ctrl_best))
        send.append('E9 seed %d: base AUC %s | best repair dAUC %+.2f pt acc loss %.2f | best ctrl dAUC %+.2f pt -> %s'
                    % (seed, f4(ba), best[0], best[1], ctrl_best, 'PASS' if passed else 'FAIL'))
    else:
        say('  seed %d GATE: incomplete' % seed)
if e9_pass:
    say('  PROMOTION: ' + ('PASS on all 3 seeds - amplification can be the headline'
                           if (len(e9_pass) == 3 and all(e9_pass))
                           else 'FAIL - report the numbers; the paper stays benchmark-led'))
say('')

say('GATE 3 - E3 Calibrate-then-Remove')
d3 = df[df.exp == 'E3']
prox = d3[d3.method == 'car_proxy']
say('  proxy rows: %d' % len(prox))
if len(prox):
    P = pd.DataFrame([parse_notes(s) for s in prox.notes])
    if 'f1' in P.columns and P.f1.notna().any():
        pb = P.loc[P.f1.idxmax()]
        say('  theta_F1 %s  tau %s  f1 %s  IoU %s'
            % (f4(pb.get('theta')), f2(pb.get('tau')), f4(pb.get('f1')), f4(pb.get('iou'))))
        if 'tau' in P.columns:
            feas = P[P.tau <= 5.0]
        else:
            feas = P.iloc[0:0]
        car = feas.loc[feas.theta.idxmax()] if len(feas) else None
        if car is not None:
            say('  theta_CaR %s  tau %s  f1 %s' % (f4(car.theta), f2(car.tau), f4(car.f1)))
        else:
            say('  theta_CaR none: no threshold reached tau <= 5 (honest negative)')
        for tag in ['f1_full', 'car_full']:
            r = one('E3', tag)
            if r is not None:
                say('  %-9s ASR %s  acc %s  n_forget %d'
                    % (tag, f2(num(r, 'asr')), f2(num(r, 'clean_acc')), int(float(r.n_forget))))
        f1r, cr = one('E3', 'f1_full'), one('E3', 'car_full')
        if f1r is not None and cr is not None:
            better = (num(cr, 'asr') is not None and num(f1r, 'asr') is not None
                      and num(cr, 'asr') <= num(f1r, 'asr')
                      and num(cr, 'clean_acc') >= num(f1r, 'clean_acc') - 2)
            say('  GATE: ' + ('PASS - CaR matches or beats the F1 choice'
                              if better else 'FAIL - CaR did not beat the F1 choice; report it'))
        send.append('E3: theta_F1 %s tau %s | theta_CaR %s tau %s | IoU %s | f1_full ASR %s acc %s | car_full ASR %s acc %s'
                    % (f4(pb.get('theta')), f2(pb.get('tau')),
                       f4(car.theta) if car is not None else 'none',
                       f2(car.tau) if car is not None else 'NA',
                       f4(pb.get('iou')),
                       f2(num(f1r, 'asr')), f2(num(f1r, 'clean_acc')),
                       f2(num(cr, 'asr')), f2(num(cr, 'clean_acc'))))
say('')

say('GATE 4 - E6 real detectors (no simulation)')
d6 = df[df.exp == 'E6']
for det in ['ac', 'ss']:
    d = d6[d6.notes.astype(str).str.contains('raw ' + det + ' flags', na=False)]
    if len(d) == 0:
        say('  %s: no rows' % det)
        continue
    r0 = d.iloc[0]
    say('  %s: flagged %d  recall %s  precision %s'
        % (det.upper(), int(float(r0.n_forget)), f4(num(r0, 'recall')), f4(num(r0, 'precision'))))
    for _, r in d.iterrows():
        say('      %-14s ASR %s  acc %s' % (r.method, f2(num(r, 'asr')), f2(num(r, 'clean_acc'))))
    send.append('E6 %s: recall %s precision %s | %s'
                % (det.upper(), f4(num(r0, 'recall')), f4(num(r0, 'precision')),
                   '; '.join('%s ASR %s' % (r.method, f2(num(r, 'asr'))) for _, r in d.iterrows())))
say('')

say('GATE 5 - E7b inoculation with synthesised trigger')
r7 = one('E7b', 'nc_synth')
if r7 is None:
    say('  missing')
else:
    n7 = parse_notes(r7.notes)
    say('  ASR %s  acc %s  IoU %s'
        % (f2(num(r7, 'asr')), f2(num(r7, 'clean_acc')), f2(n7.get('IoU'))))
    send.append('E7b: ASR %s acc %s IoU %s'
                % (f2(num(r7, 'asr')), f2(num(r7, 'clean_acc')), f2(n7.get('IoU'))))
say('')


def find(p):
    for d in ('.', W):
        q = os.path.join(d, p)
        if os.path.exists(q):
            return q
    return None


def sha12(p):
    q = find(p)
    return hashlib.sha256(open(q, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:12] if q else 'missing'


meta = dict(
    person='sai', seed=0, notebook='fml-sai-main',
    generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    host=platform.platform(), python=platform.python_version(),
    torch=torch.__version__, cuda=torch.version.cuda,
    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    code_sha12={f: sha12(f) for f in ('core_kaggle.py', 'person_sai.py') if find(f)},
    csv='runs_sai.csv', rows=int(len(df)),
    rows_per_exp={e: int((df.exp == e).sum()) for e in ['E1', 'E9', 'E3', 'E6', 'E7b']},
    duplicate_keys=dup,
    audit=('PASS' if not problems else 'FAIL'),
    block_status=RUN_STATUS if 'RUN_STATUS' in globals() else {},
    gates=GATES,
)

open(os.path.join(W, 'GATE_SUMMARY.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
json.dump(meta, open(os.path.join(W, 'meta.json'), 'w'), indent=2)
json.dump(GATES, open(os.path.join(W, 'gate_summary.json'), 'w'), indent=2)

E1_METHODS = ['retrain_delF', 'finetune_retain', 'negrad_plus', 'scrub', 'recipe']
e1s = df[(df.exp == 'E1') & df.method.isin(E1_METHODS)]
if len(e1s):
    for val, name in [('asr', 'E1_pivot_asr.csv'), ('clean_acc', 'E1_pivot_acc.csv')]:
        e1s.pivot_table(index=['recall', 'precision'], columns='method',
                        values=val, aggfunc='first').round(2).to_csv(os.path.join(W, name))

zpath = os.path.join(W, 'FML_Sai_results.zip')
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(CSV, 'runs_sai.csv')
    zf.write(os.path.join(W, 'GATE_SUMMARY.txt'), 'GATE_SUMMARY.txt')
    zf.write(os.path.join(W, 'gate_summary.json'), 'gate_summary.json')
    zf.write(os.path.join(W, 'meta.json'), 'meta.json')
    for f in ('core_kaggle.py', 'person_sai.py', 'E1_pivot_asr.csv', 'E1_pivot_acc.csv'):
        p = find(f)
        if p is not None:
            zf.write(p, f)

say('')
say('=' * 64)
say('PASTE THIS TO THE OWNER')
say('=' * 64)
for s in send:
    say(s)
say('')
say('zip: %s  (%.0f KB)' % (zpath, os.path.getsize(zpath) / 1024))
print('\nDownload it from the right-hand Output pane: FML_Sai_results.zip')
"""))

CELLS.append(md(r"""## Send back

1. **`FML_Sai_results.zip`** — Output pane (right side) → `FML_Sai_results.zip` → download. It contains `runs_sai.csv`, `GATE_SUMMARY.txt`, `gate_summary.json`, `meta.json`, the two frozen source files, and the E1 pivots.
2. **The `PASTE THIS TO THE OWNER` block** printed by Cell 8 — copy it into the message.

Do not hand-edit `runs_sai.csv`. If a block failed, send `GATE_SUMMARY.txt` and the traceback from the log; the CSV keeps every row that was measured before the failure.

**If the session dies early:** download `/kaggle/working/runs_sai.csv`, upload it to the private dataset `fml-runs-sai`, attach the dataset to a new session, and run Cells 1 → 7 again. Completed rows are skipped by the resume key, so only the unfinished part re-runs.
"""))


note = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTDIR, 'FML_Sai_Kaggle_Run.ipynb')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', encoding='utf-8', newline='\n') as fh:
    json.dump(note, fh, ensure_ascii=False, indent=1)
    fh.write('\n')

# ------------------------------------------------------------------ validate
raw = open(out, encoding='utf-8').read()
nb = json.loads(raw)
assert nb['nbformat'] == 4 and len(nb['cells']) == len(CELLS)
ncode = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    ncode += 1
    text = ''.join(c['source'])
    if text.startswith('%%writefile'):
        head, body = text.split('\n', 1)
        compile(body, 'cell%d' % i, 'exec')
        print('cell %d: %s -> %d lines, sha %s'
              % (i, head.strip(), body.count('\n'), sha12(body)))
    else:
        compile(text, 'cell%d' % i, 'exec')
print('OK  %s' % out)
print('cells: %d (%d code, %d markdown), bytes: %d'
      % (len(nb['cells']), ncode, len(nb['cells']) - ncode, len(raw.encode('utf-8'))))
