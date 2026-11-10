"""Generate FML_Divyansh_Fix.ipynb.

Re-measures Divyansh's 21 invalid rows (base_eval + 10 negrad_plus + 10 scrub)
and optionally adds the stronger SISA baseline at 40 epochs/shard. The frozen
core is embedded byte-for-byte and hash-checked; broken functions are rebound
in memory only.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

TEAM = Path(r'E:\FML\team')
NB_DIR = Path(r'E:\FML\handouts\notebooks')
SRC_NB = NB_DIR / 'FML_Divyansh_Kaggle_Run.ipynb'
OUT_NB = NB_DIR / 'FML_Divyansh_Fix.ipynb'
SRC_CSV = Path(r'E:\FML\results\divyansh_2026-11-03\runs_divyansh.csv')

core = (TEAM / 'core_kaggle.py').read_bytes().replace(b'\r\n', b'\n').decode('utf-8')
CORE_SHA = hashlib.sha256(core.encode('utf-8')).hexdigest()
meta = json.loads(SRC_NB.read_text(encoding='utf-8'))['metadata']

src = pd.read_csv(SRC_CSV)
sel = src[(src.exp == 'E1') & (src.method.isin(['negrad_plus', 'scrub']))].copy()
sel['recall'] = sel['recall'].round(4)
sel['precision'] = sel['precision'].round(4)
sel = sel.sort_values(['method', 'recall', 'precision'])
old_lines = []
for r in sel.itertuples():
    old_lines.append("    (%r, %r, %r, %r): (%r, %r),"
                     % (r.exp, r.method, float(r.recall), float(r.precision),
                        float(r.clean_acc), float(r.asr)))
OLD_DICT = 'OLD = {\n' + '\n'.join(old_lines) + '\n}'

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'id': 'c%d' % len(cells), 'metadata': {},
                  'source': text.strip('\n').splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'id': 'c%d' % len(cells), 'execution_count': None,
                  'metadata': {}, 'outputs': [],
                  'source': text.strip('\n').splitlines(keepends=True)})


def code_exact(text):
    cells.append({'cell_type': 'code', 'id': 'c%d' % len(cells), 'execution_count': None,
                  'metadata': {}, 'outputs': [], 'source': text.splitlines(keepends=True)})


md(r"""
# FML - Divyansh completion run (seed 2)

**One session, no attached datasets.** Re-measures the 21 invalid rows in `runs_divyansh.csv` and optionally adds the stronger SISA baseline.

### What is re-measured
| Block | Rows | Why it is invalid |
|---|---|---|
| `base_eval` | 1 | the poisoned base model's accuracy/ASR was never logged |
| E1 `negrad_plus` | 10 | only 6-21 optimizer steps at lr 1e-3; ASR 0 came with a 3-12 point accuracy drop |
| E1 `scrub` | 10 | constant lr=0.01 for ~279 steps, no schedule; accuracy fell to 51.41% |
| **Invalid total** | **21** | |
| SISA `sisa5_e40` | 2 (optional) | stronger baseline: 40 epochs/shard instead of the 12 used in the first run |

### What changed
* `negrad_plus`: one pass over the **retain** set per epoch (same 3-epoch budget as the other methods), **OneCycle** schedule, and BN statistics fitted on **retain only**.
* `scrub`: same OneCycle schedule and BN guard; the teacher is still `finetune_retain`.
* SISA (optional): K=5 random shards, **40 epochs/shard**, shards containing flags retrained from scratch, majority-vote ensemble. The original 12-epoch rows stay valid; this adds a stronger baseline under the same key system.

### What is NOT touched
`core_kaggle.py` is written byte-identical to the frozen file (sha256 checked) and `negrad_plus` / `scrub_finetune` are rebound **in memory**. `retrain_delF`, `finetune_retain`, `recipe`, `nc_style`, all E7/E7b rows and the existing 12-epoch SISA rows stay as they are.

### Settings
* Accelerator: **GPU T4 x 2**
* Internet: **On** (PyTorch downloads CIFAR-10)
* Datasets: **none**
* Runtime: **~1 h 45 min** for the 21 rows; **~2 h 50 min** with `RUN_SISA = True`. Set `RUN_SISA = False` to skip the two SISA rows.

### Output
`FML_Divyansh_Fix.zip` -> `runs_divyansh_v2.csv`, `divyansh_fix_meta.json`. Download and hand it back.

**Resume.** Every finished row is flushed to `/kaggle/working/runs_divyansh_v2.csv` immediately, so a cell crash does not lose completed rows; re-running Cell 7 in the same live session skips them. A full session restart begins with an empty working directory.
""")

code(r"""
# CELL 1 - knobs and environment check. Nothing heavy happens here.
import os, sys, json, time, platform, hashlib, zipfile, shutil

RUN_E1FIX = True    # 21 rows (base_eval + 10 negrad_plus + 10 scrub): ~100 min
RUN_SISA  = True    # 2 extra SISA rows at 40 epochs/shard: +~100 min
SISA_EPOCHS = 40

print('python  ', sys.version.split()[0])
print('platform', platform.platform())
try:
    import torch
    print('torch   ', torch.__version__, '| cuda', torch.cuda.is_available(),
          '| gpus', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print('   gpu%d  %s %.1f GB' % (i, torch.cuda.get_device_name(i), p.total_memory / 1e9))
except Exception as e:
    print('torch import failed:', e)
print('cwd     ', os.getcwd())
print('kaggle  ', os.path.exists('/kaggle/working'))
print('plan    ', 'E1fix=%s SISA=%s' % (RUN_E1FIX, RUN_SISA))
""")

code_exact('%%writefile core_kaggle.py\n' + core)

code(r"""
# CELL 3 - the frozen core must arrive byte-identical
EXPECT = '__SHA__'
raw = open('core_kaggle.py', 'rb').read()
got = hashlib.sha256(raw).hexdigest()
assert got == EXPECT, 'core_kaggle.py hash mismatch: %s != %s' % (got, EXPECT)
print('core_kaggle.py  %d bytes  sha256[:12]=%s  OK' % (len(raw), got[:12]))
""".replace('__SHA__', CORE_SHA))

code(r"""
# CELL 4 - load the frozen core into this notebook, then CIFAR-10
src = open('core_kaggle.py').read()
exec(compile(src, 'core_kaggle.py', 'exec'))
print('device', dev, '| gpus', NGPU, '| BS', BS, '| epochs', EPOCHS)
load_cifar()
""")

code(r"""
# CELL 5 - corrected baselines, rebound over the frozen ones, plus SISA helpers.
from tqdm.auto import tqdm
import math

def _cycle_opt(model, lr, total_steps):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(total_steps, 1), pct_start=0.3)
    return opt, sched

def _pick(flag, n, salt):
    rng = np.random.RandomState((SEED + 7) * 131 + salt)
    return rng.choice(flag, size=n, replace=True)

def negrad_plus_fix(model0, X, y, Fset, epochs=3, lr=1e-3, bs=None):
    # Ascent on forget + descent on retain, one pass over retain per epoch.
    # BN statistics are fitted on the retain stream only: the forget set is
    # corrupt and must not define the normalisation used at evaluation.
    bs = bs or BS
    retain = np.where(~Fset)[0]
    flag = np.where(Fset)[0]
    model = clone_model(model0)
    opt, sched = _cycle_opt(model, lr, epochs * math.ceil(len(retain) / bs))
    step = 0
    for ep in range(epochs):
        perm = np.random.RandomState(SEED * 977 + ep).permutation(retain)
        for k in range(0, len(perm), bs):
            ri = torch.from_numpy(perm[k:k + bs]).to(dev)
            fi = torch.from_numpy(_pick(flag, len(ri), step)).to(dev)
            model.train()
            loss_r = F.cross_entropy(model(X[ri]), y[ri])
            model.eval()
            loss_f = -F.cross_entropy(model(X[fi]), y[fi])
            loss = loss_r + loss_f
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
    model.eval()
    cleanup()
    return model

def scrub_finetune_fix(model0, X, y, Fset, epochs=3, lr=0.01, bs=None, teach_epochs=3):
    # SCRUB with a proper OneCycle schedule and the same BN guard.
    # The frozen version used a constant lr=0.01 for ~279 steps, which destroyed
    # clean accuracy instead of unlearning the backdoor.
    bs = bs or BS
    retain = np.where(~Fset)[0]
    flag = np.where(Fset)[0]
    teacher = finetune_retain(model0, X, y, Fset, epochs=teach_epochs, lr=lr)
    teacher.eval()
    model = clone_model(model0)
    opt, sched = _cycle_opt(model, lr, epochs * math.ceil(len(retain) / bs))
    for ep in range(epochs):
        perm = np.random.RandomState(SEED * 613 + ep).permutation(retain)
        for k in range(0, len(perm), bs):
            ri = torch.from_numpy(perm[k:k + bs]).to(dev)
            fi = torch.from_numpy(_pick(flag, len(ri), ep * 1000 + k)).to(dev)
            model.train()
            loss_r = F.cross_entropy(model(augment_batch(X[ri])), y[ri])
            model.eval()
            with torch.no_grad():
                t = teacher(X[fi]).softmax(1)
            loss_f = F.kl_div(model(X[fi]).log_softmax(1), t, reduction='batchmean')
            loss = loss_r + loss_f
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    cleanup()
    return model

def sisa_train(Xp, yp, F, K=5, epochs=40):
    n = len(yp)
    shards = np.array_split(np.random.RandomState(SEED).permutation(n), K)
    models = []
    for j, ids in enumerate(tqdm(shards, desc='SISA shard', unit='shard')):
        m = CIFARModel('resnet18').to(dev)
        m, _ = train_model(Xp[ids], yp[ids], model=m, epochs=epochs, tag='sisa_b%d' % j)
        if F[ids].any():
            keep = ids[~F[ids]]
            m = CIFARModel('resnet18').to(dev)
            m, _ = train_model(Xp[keep], yp[keep], model=m, epochs=epochs, tag='sisa_r%d' % j)
        m.cpu().eval()
        models.append(m)
        cleanup()
    return models

@torch.no_grad()
def sisa_eval(models, attack):
    dm = [m.to(dev) for m in models]
    prob = sum(predict_probs(m, Xte) for m in dm) / len(dm)
    acc = (prob.argmax(1) == yte).float().mean().item() * 100
    nt = (yte != TARGET)
    xt = add_trigger(Xte[nt], attack)
    pt = sum(predict_probs(m, xt) for m in dm) / len(dm)
    asr = (pt.argmax(1) == TARGET).float().mean().item() * 100
    return round(acc, 2), round(asr, 2)

negrad_plus = negrad_plus_fix
scrub_finetune = scrub_finetune_fix
print('rebound negrad_plus    ->', negrad_plus.__name__)
print('rebound scrub_finetune ->', scrub_finetune.__name__)
print('SISA epochs per shard  ->', SISA_EPOCHS)
print('tqdm', __import__('tqdm').__version__)
""")

code(r"""
# CELL 6 - shared runner. Writes a NEW csv; the first-run file is untouched.
def run_e1_fix(LOG, base, Xp, yp, ip):
    arch, attack, rate = 'resnet18', 'badnets', 0.05
    for (r, p) in tqdm(E1_CONFIGS, desc='E1-fix cell', unit='cell'):
        F, st = make_forget_set(ip, yp, r, p, SEED)
        for meth in ['negrad_plus', 'scrub']:
            key = dict(exp='E1', arch=arch, attack=attack, rate=rate, seed=SEED,
                       recall=st['recall'], precision=st['precision'],
                       n_forget=st['n_forget'], method=meth, detector='none')
            if LOG.has(**key):
                continue
            t0 = time.time()
            m, _ = apply_method(meth, base, Xp, yp, F, attack)
            log_removal(LOG, m, 'E1', arch, attack, rate, meth,
                        st['recall'], st['precision'], st['n_forget'],
                        time.time() - t0,
                        notes='v2 fix: one pass over retain/epoch, OneCycle, BN stats from retain only')
            del m
            cleanup()

def run_sisa(LOG, Xp, yp, ip):
    arch, attack, rate = 'resnet18', 'badnets', 0.05
    for r in tqdm([1.0, 0.5], desc='SISA cell', unit='cell'):
        F, st = make_forget_set(ip, yp, r, 1.0, SEED)
        key = dict(exp='SISA', arch=arch, attack=attack, rate=rate, seed=SEED,
                   recall=st['recall'], precision=st['precision'],
                   n_forget=st['n_forget'],
                   method='sisa5_e%d' % SISA_EPOCHS, detector='none')
        if LOG.has(**key):
            continue
        t0 = time.time()
        models = sisa_train(Xp, yp, F, K=5, epochs=SISA_EPOCHS)
        acc, asr = sisa_eval(models, attack)
        dt = time.time() - t0
        LOG.log(**key, clean_acc=acc, asr=asr, time_sec=dt,
                notes='K=5 random shards, %d epochs/shard, retrain shards containing flags'
                      % SISA_EPOCHS)
        print('  SISA recall %.2f: acc %.2f  asr %.2f  %.1f min'
              % (st['recall'], acc, asr, dt / 60))
        del models
        cleanup()

def run_all():
    setup('divyansh', 2)
    LOG = Logger('runs_divyansh_v2.csv')
    arch, attack, rate = 'resnet18', 'badnets', 0.05
    t_all = time.time()

    print('\n===== poisoned base model =====')
    base, Xp, yp, ip, bt = poison_and_train(arch, attack, rate, 2)
    n_poison = int(ip.sum())
    acc0, asr0 = evaluate(base, attack)
    print('  base: acc %.2f  asr %.2f  (n_poison %d)' % (acc0, asr0, n_poison))

    key = dict(exp='E1', arch=arch, attack=attack, rate=rate, seed=2,
               recall=1.0, precision=1.0, n_forget=n_poison,
               method='base_eval', detector='none')
    if not LOG.has(**key):
        LOG.log(**key, clean_acc=acc0, asr=asr0, time_sec=bt,
                notes='poisoned base model; seed 2; Table 1 starting point')

    if RUN_E1FIX:
        print('\n===== E1 baselines (21 rows) =====')
        run_e1_fix(LOG, base, Xp, yp, ip)
    else:
        print('\nRUN_E1FIX is False - skipping the 20 baseline rows')

    if RUN_SISA:
        print('\n===== SISA (2 rows, %d epochs/shard) =====' % SISA_EPOCHS)
        run_sisa(LOG, Xp, yp, ip)
    else:
        print('\nRUN_SISA is False - skipping SISA')

    del base, Xp
    cleanup()
    print('\nall requested rows done in %.1f min' % ((time.time() - t_all) / 60))
""")

code(r"""
# CELL 7 - the long cell
run_all()
""")

code(r"""
# CELL 8 - audit, before/after table, package
import pandas as pd

PATH = 'runs_divyansh_v2.csv'
__OLD__

def audit(path):
    if not os.path.exists(path):
        print('MISSING', path)
        return False
    d = pd.read_csv(path)
    key_cols = ['exp', 'arch', 'attack', 'rate', 'seed', 'recall',
                'precision', 'n_forget', 'method', 'detector']
    dup = int(d.duplicated(subset=key_cols).sum())
    print('\n--- %s ---' % path)
    print('rows %d | columns %d | duplicate keys %d' % (len(d), d.shape[1], dup))
    for exp in sorted(d.exp.unique()):
        print('  %-5s %d rows' % (exp, int((d.exp == exp).sum())))
    bad = d[d.clean_acc.isna() | d.asr.isna()]
    print('missing acc/asr %d | time_sec<=0 %d' % (len(bad), int((d.time_sec <= 0).sum())))

    print('\n%-26s %8s %8s %8s %8s %8s' %
          ('exp/method/cell', 'old_acc', 'old_asr', 'new_acc', 'new_asr', 'dASR'))
    for row in d.itertuples():
        k = (row.exp, row.method, round(float(row.recall), 4), round(float(row.precision), 4))
        old = OLD.get(k)
        oa, oas = old if old else (float('nan'), float('nan'))
        da = row.asr - oas if old else float('nan')
        print('%-26s %8.2f %8.2f %8.2f %8.2f %8.2f' %
              ('%s/%s/r%.2f' % (row.exp, row.method, row.recall), oa, oas,
               row.clean_acc, row.asr, da))

    expected = 1 + (20 if RUN_E1FIX else 0) + (2 if RUN_SISA else 0)
    ok = (len(d) == expected) and dup == 0 and len(bad) == 0
    print('\nexpected %d rows -> %s' % (expected, 'PASS' if ok else 'CHECK ABOVE'))
    return ok

ok = audit(PATH)

meta = dict(
    purpose='re-measure 21 invalid divyansh rows + SISA e%d (seed 2)' % SISA_EPOCHS,
    core_sha256=EXPECT,
    rows_expected=1 + (20 if RUN_E1FIX else 0) + (2 if RUN_SISA else 0),
    sisa_epochs=SISA_EPOCHS if RUN_SISA else None,
    fixes=['negrad_plus one-pass+OneCycle+BN-retain',
           'scrub OneCycle+BN-retain'],
    generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    host=platform.platform(),
    files={PATH: int(len(pd.read_csv(PATH)))},
)
json.dump(meta, open('divyansh_fix_meta.json', 'w'), indent=2)

with zipfile.ZipFile('FML_Divyansh_Fix.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in [PATH, 'divyansh_fix_meta.json']:
        z.write(f)

print('\n============================================================')
print('PASTE THIS BACK')
print('============================================================')
print('runs_divyansh_v2.csv  %d rows' % len(pd.read_csv(PATH)))
print('zip: /kaggle/working/FML_Divyansh_Fix.zip')
print('============================================================')
""".replace('__OLD__', OLD_DICT))

nb = {
    'cells': cells,
    'metadata': meta,
    'nbformat': 4,
    'nbformat_minor': 5,
}
OUT_NB.parent.mkdir(parents=True, exist_ok=True)
OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('wrote', OUT_NB, OUT_NB.stat().st_size, 'bytes')
print('core sha256', CORE_SHA)
print('cells', len(cells))
print('old rows embedded', len(sel))
