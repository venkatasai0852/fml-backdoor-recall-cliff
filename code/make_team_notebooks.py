# -*- coding: utf-8 -*-
"""Build the Sarang / Divyansh / Urja Kaggle notebooks from the validated Sai notebook.

The Sai notebook (FML_Sai_Kaggle_Run.ipynb) was executed end-to-end on 2x T4 and
produced a passing audit. This generator reuses its cells 1-8 verbatim wherever the
text is person-independent, and replaces only: the driver payload, the setup() call,
the block list / CALLS, the row expectations, the gates, and the file names.

Usage:  python make_team_notebooks.py
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = r'E:\FML\handouts\notebooks'
BASE_NB = os.path.join(OUTDIR, 'FML_Sai_Kaggle_Run.ipynb')
CORE = os.path.join(HERE, 'core_kaggle.py')

CORE_SHA = 'ad7a363926a5'


def read_lf(path):
    b = open(path, 'rb').read().replace(b'\r\n', b'\n')
    t = b.decode('utf-8')
    if not t.endswith('\n'):
        t += '\n'
    return t


def sha12(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


def lines(s):
    return s.splitlines(keepends=True)


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines(s)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(s)}


# ------------------------------------------------------------------ person configs

SARANG_EXTRA = r"""say('GATE 2 - E7 VGG-16-BN (BadNets and Blend)')
d7 = df[df.exp == 'E7']


def e7row(attack, method, rec):
    d = d7[(d7.attack == attack) & (d7.method == method)]
    d = d[np.isclose(pd.to_numeric(d.recall, errors='coerce'), rec, atol=2e-3)]
    return d.iloc[0] if len(d) else None


if len(d7) == 0:
    say('  no E7 rows yet')
for attack in ['badnets', 'blend']:
    g = e7row(attack, 'gold', 1.0)
    if g is None:
        say('  %-8s missing' % attack)
        continue
    say('  %-8s gold: ASR %s  acc %s' % (attack, f2(num(g, 'asr')), f2(num(g, 'clean_acc'))))
    send.append('E7 %s gold: ASR %s acc %s'
                % (attack, f2(num(g, 'asr')), f2(num(g, 'clean_acc'))))
    for rec in [0.75, 0.50]:
        rt = e7row(attack, 'retrain_delF', rec)
        rp = e7row(attack, 'recipe', rec)
        say('    recall %.2f  retrain ASR %s acc %s | recipe ASR %s acc %s'
            % (rec, f2(num(rt, 'asr')), f2(num(rt, 'clean_acc')),
               f2(num(rp, 'asr')), f2(num(rp, 'clean_acc'))))
        send.append('E7 %s r=%.2f: retrain ASR %s | recipe ASR %s'
                    % (attack, rec, f2(num(rt, 'asr')), f2(num(rp, 'asr'))))
gacc = [num(e7row(a, 'gold', 1.0), 'clean_acc') for a in ['badnets', 'blend']
        if e7row(a, 'gold', 1.0) is not None]
if gacc:
    ok = min([x for x in gacc if x is not None] or [0]) >= 90
    GATES['E7_vgg_gold'] = dict(gold_acc=[x for x in gacc if x is not None], pass_=bool(ok))
    say('  GATE: ' + ('PASS - VGG gold trained' if ok else 'CHECK - VGG gold below 90 acc'))
say('')

say('GATE 3 - E8 ascent-strength cliff')
d8 = df[df.exp == 'E8']
if len(d8) == 0:
    say('  no E8 rows yet')
for _, r in d8.sort_values('method').iterrows():
    fn = parse_notes(r.notes).get('feature_norm')
    say('  %-26s ASR %s  acc %s  feature_norm %s'
        % (r.method, f2(num(r, 'asr')), f2(num(r, 'clean_acc')), f2(fn)))
if len(d8):
    send.append('E8: ' + '; '.join('%s ASR %s acc %s'
                                   % (r.method, f2(num(r, 'asr')), f2(num(r, 'clean_acc')))
                                   for _, r in d8.iterrows()))
say('')
"""

DIVYANSH_EXTRA = r"""say('GATE 2 - E7 Blend on ResNet-18')
d7 = df[df.exp == 'E7']


def e7row(method, rec):
    d = d7[d7.method == method]
    d = d[np.isclose(pd.to_numeric(d.recall, errors='coerce'), rec, atol=2e-3)]
    return d.iloc[0] if len(d) else None


g = e7row('gold', 1.0)
if g is None:
    say('  no E7 rows yet')
else:
    say('  gold: ASR %s  acc %s' % (f2(num(g, 'asr')), f2(num(g, 'clean_acc'))))
    send.append('E7 blend gold: ASR %s acc %s'
                % (f2(num(g, 'asr')), f2(num(g, 'clean_acc'))))
    for rec in [0.75, 0.50]:
        rt = e7row('retrain_delF', rec)
        rp = e7row('recipe', rec)
        say('    recall %.2f  retrain ASR %s acc %s | recipe ASR %s acc %s'
            % (rec, f2(num(rt, 'asr')), f2(num(rt, 'clean_acc')),
               f2(num(rp, 'asr')), f2(num(rp, 'clean_acc'))))
        send.append('E7 blend r=%.2f: retrain ASR %s | recipe ASR %s'
                    % (rec, f2(num(rt, 'asr')), f2(num(rp, 'asr'))))
say('')

say('GATE 3 - E7b synthesised-trigger inoculation')
r7 = one('E7b', 'nc_synth')
if r7 is None:
    say('  missing')
else:
    mm = parse_notes(r7.notes).get('mask_mass')
    say('  ASR %s  acc %s  mask_mass %s'
        % (f2(num(r7, 'asr')), f2(num(r7, 'clean_acc')), f2(mm)))
    send.append('E7b: ASR %s acc %s mask_mass %s'
                % (f2(num(r7, 'asr')), f2(num(r7, 'clean_acc')), f2(mm)))
say('')

say('GATE 4 - SISA K=5')
ds = df[df.exp == 'SISA']
if len(ds) == 0:
    say('  no SISA rows yet')
for rec in [1.0, 0.50]:
    d = ds[np.isclose(pd.to_numeric(ds.recall, errors='coerce'), rec, atol=2e-3)]
    if len(d):
        r = d.iloc[0]
        say('  recall %.2f  ASR %s  acc %s'
            % (rec, f2(num(r, 'asr')), f2(num(r, 'clean_acc'))))
        send.append('SISA r=%.2f: ASR %s acc %s'
                    % (rec, f2(num(r, 'asr')), f2(num(r, 'clean_acc'))))
say('')
"""

URJA_EXTRA = r"""say('GATE 1 - E2 residual-poison cliff (3 seeds)')
d2 = df[df.exp == 'E2'].copy()
d2['residual'] = pd.to_numeric(
    d2.notes.astype(str).str.extract(r'residual=(\d+)')[0], errors='coerce')
if len(d2) == 0:
    say('  no E2 rows yet')
for seed in sorted(d2.seed.dropna().unique()):
    d = d2[d2.seed == seed]
    say('  seed %d' % int(seed))
    for nres in [2500, 1250, 625, 312, 156, 78, 39, 20, 10, 0]:
        rt = d[(d.residual == nres) & (d.method == 'retrain_delF')]
        rp = d[(d.residual == nres) & (d.method == 'recipe')]
        if len(rt) == 0 and len(rp) == 0:
            continue
        say('    residual %4d  retrain ASR %s acc %s | recipe ASR %s acc %s'
            % (nres,
               f2(num(rt.iloc[0], 'asr')) if len(rt) else 'NA',
               f2(num(rt.iloc[0], 'clean_acc')) if len(rt) else 'NA',
               f2(num(rp.iloc[0], 'asr')) if len(rp) else 'NA',
               f2(num(rp.iloc[0], 'clean_acc')) if len(rp) else 'NA'))
    r2500 = d[(d.residual == 2500) & (d.method == 'retrain_delF')]
    r0 = d[(d.residual == 0) & (d.method == 'retrain_delF')]
    if len(r2500) and len(r0):
        a25 = num(r2500.iloc[0], 'asr')
        a0 = num(r0.iloc[0], 'asr')
        ok = (a25 is not None and a0 is not None and a25 >= 90 and a0 <= 10)
        GATES['E2_sanity_seed%d' % int(seed)] = dict(
            asr_2500=a25, asr_0=a0, pass_=bool(ok))
        say('  SEED %d GATE: %s  (residual 2500 ASR %s, residual 0 ASR %s)'
            % (int(seed), 'PASS' if ok else 'CHECK', f2(a25), f2(a0)))
        send.append('E2 seed %d: residual 2500 ASR %s | residual 0 ASR %s -> %s'
                    % (int(seed), f2(a25), f2(a0), 'PASS' if ok else 'CHECK'))
say('')

say('GATE 2 - E4 recipe ablations')
d4 = df[df.exp == 'E4']
if len(d4) == 0:
    say('  no E4 rows yet')
seen = []
for _, row in d4.iterrows():
    k = (round(float(row.recall), 4), round(float(row.precision), 4))
    if k not in seen:
        seen.append(k)
for (rec, prec) in seen:
    d = d4[np.isclose(pd.to_numeric(d4.recall, errors='coerce'), rec, atol=2e-3)
           & np.isclose(pd.to_numeric(d4.precision, errors='coerce'), prec, atol=2e-3)]
    say('  cell (%.2f, %.2f)' % (rec, prec))
    for src in ['penult', 'early', 'refmodel']:
        r = d[(d.method == 'label_acc') & (d.detector == src)]
        if len(r):
            say('    label_acc %-8s %s'
                % (src, f4(parse_notes(r.iloc[0].notes).get('label_acc'))))
    for meth in ['recipe', 'recipe_decouple', 'recipe_noinnoc',
                 'recipe_early', 'recipe_refmodel']:
        r = d[d.method == meth]
        if len(r):
            say('    %-18s ASR %s  acc %s'
                % (meth, f2(num(r.iloc[0], 'asr')), f2(num(r.iloc[0], 'clean_acc'))))
    send.append('E4 (%.2f,%.2f): recipe ASR %s | noinnoc ASR %s | decouple ASR %s'
                % (rec, prec,
                   f2(num(d[d.method == 'recipe'].iloc[0], 'asr')) if len(d[d.method == 'recipe']) else 'NA',
                   f2(num(d[d.method == 'recipe_noinnoc'].iloc[0], 'asr')) if len(d[d.method == 'recipe_noinnoc']) else 'NA',
                   f2(num(d[d.method == 'recipe_decouple'].iloc[0], 'asr')) if len(d[d.method == 'recipe_decouple']) else 'NA'))
say('')

say('GATE 3 - E5 relearning + MIA')
d5 = df[df.exp == 'E5']
if len(d5) == 0:
    say('  no E5 rows yet')
steps = {}
for tag in ['poisoned', 'recipe', 'gold']:
    rr = d5[(d5.method == 'relearn') & (d5.detector == tag)]
    mi = d5[(d5.method == 'mia') & (d5.detector == tag)]
    if len(rr) == 0 and len(mi) == 0:
        continue
    st = parse_notes(rr.iloc[0].notes).get('steps_to_asr90') if len(rr) else None
    au = num(mi.iloc[0], 'auc') if len(mi) else None
    steps[tag] = st
    say('  %-8s relearn_steps %s  MIA AUC %s' % (tag, f2(st), f4(au)))
    send.append('E5 %s: relearn_steps %s MIA AUC %s' % (tag, f2(st), f4(au)))
if 'poisoned' in steps and 'recipe' in steps and steps['poisoned'] and steps['recipe']:
    say('  suppression check: recipe / poisoned relearn ratio %.2f  (>0.8 means '
        'removal is suppression, not elimination)' % (steps['recipe'] / steps['poisoned']))
say('')
"""

CONFIGS = {
    'sarang': dict(
        key='sarang', name='Sarang', seed=1, seed_label='1',
        driver='person_sarang.py', sha='264695b6fdc6',
        csv='runs_sarang.csv', zip='FML_Sarang_results.zip',
        snap='FML_Sarang_results_partial.zip', notebook='fml-sarang-main',
        blocks_js="['E1', 'E7', 'E8']",
        calls="    'E1':  lambda: run_e1(LOG, 1),\n"
              "    'E7':  lambda: run_e7_vgg(),\n"
              "    'E8':  lambda: run_e8(LOG, 1),",
        pre7=("VGG_BS = globals().get('VGG_BS', 256)\n\n\n"
              "def run_e7_vgg():\n"
              "    # E7 is VGG-16-BN; E1 and E8 stay at the protocol batch 512.\n"
              "    global BS\n"
              "    old = BS\n"
              "    BS = VGG_BS\n"
              "    try:\n"
              "        run_e7(LOG, 1, attacks=('badnets', 'blend'))\n"
              "    finally:\n"
              "        BS = old\n"),
        expect={'E1': (53, 53), 'E7': (10, 10), 'E8': (6, 6)},
        has_e1=True, extra=SARANG_EXTRA, rows=69,
        wall='5.5-6 h', max_hours=8.0,
        vgg=True,
        intro=("**E1** (main detector grid, seed 1), **E7** (VGG-16-BN x BadNets/Blend), "
               "**E8** (ascent-strength cliff). Target: **69 rows** in `runs_sarang.csv`."),
        notes=("E7 trains VGG-16-BN. Cell 6 checks that a batch of 256 fits and Cell 7 "
               "applies that batch only to E7; E1 and E8 keep the protocol batch 512, so "
               "seed 1 stays comparable with seeds 0 and 2."),
        times="E1 2.5-3 h &middot; E7 2-2.5 h &middot; E8 ~30 min",
    ),
    'divyansh': dict(
        key='divyansh', name='Divyansh', seed=2, seed_label='2',
        driver='person_divyansh.py', sha='7fcc1c52341a',
        csv='runs_divyansh.csv', zip='FML_Divyansh_results.zip',
        snap='FML_Divyansh_results_partial.zip', notebook='fml-divyansh-main',
        blocks_js="['E1', 'E7B', 'SISA']",
        calls="    'E1':  lambda: run_e1(LOG, 2),\n"
              "    'E7B': lambda: run_e7b(LOG, 2),\n"
              "    'SISA': lambda: run_sisa(LOG, 2),",
        pre7='',
        expect={'E1': (53, 53), 'E7': (5, 5), 'E7b': (1, 1), 'SISA': (2, 2)},
        has_e1=True, extra=DIVYANSH_EXTRA, rows=61,
        wall='4.5-5 h', max_hours=8.0,
        vgg=False,
        intro=("**E1** (main detector grid, seed 2), **E7B** (Blend on ResNet-18 + trigger "
               "synthesis), **SISA** (K=5 shards). Target: **61 rows** in `runs_divyansh.csv`."),
        notes=("SISA trains five 12-epoch shard models per recall setting; it is the "
               "slowest block after E1. The trigger-synthesis cache is reused if you resume."),
        times="E1 2.5-3 h &middot; E7B ~50 min &middot; SISA ~45 min",
    ),
    'urja': dict(
        key='urja', name='Urja', seed=0, seed_label='0',
        driver='person_urja.py', sha='a10bffc441b0',
        csv='runs_urja.csv', zip='FML_Urja_results.zip',
        snap='FML_Urja_results_partial.zip', notebook='fml-urja-main',
        blocks_js="['E2', 'E4', 'E5']",
        calls="    'E2':  lambda: run_e2(LOG, seeds=(0, 1, 2)),\n"
              "    'E4':  lambda: run_e4(LOG, 0),\n"
              "    'E5':  lambda: run_e5(LOG, 0),",
        pre7='',
        expect={'E2': (60, 60), 'E4': (32, 32), 'E5': (6, 6)},
        has_e1=False, extra=URJA_EXTRA, rows=98,
        wall='7-7.5 h', max_hours=8.0,
        vgg=False,
        intro=("**E2** (residual-poison cliff, 3 seeds), **E4** (recipe ablations), "
               "**E5** (relearning + MIA). Target: **98 rows** in `runs_urja.csv`."),
        notes=("E2 alone is about 5 h. Plan for two sessions: run Cells 1-7 now; if the "
               "session ends before E4/E5, download the CSV, attach it as `fml-runs-urja`, "
               "and run Cells 1 &rarr; 7 again (Cells 3-4 rewrite the frozen files). "
               "Finished E2 rows skip in seconds."),
        times="E2 ~5 h &middot; E4 1.5-2 h &middot; E5 ~30 min",
    ),
}

# ------------------------------------------------------------------ cell builders

CELL7_TMPL = r'''# CELL 7 - run all blocks. Long: {wall}. Safe to re-run; completed rows are skipped.
import os, time, zipfile, traceback

BLOCKS = {blocks}
MAX_HOURS = {max_hours}          # headroom under Kaggle's ~9 h GPU session cap
{pre}
CALLS = {{
{calls}
}}

if 'boot' not in globals():
    raise RuntimeError('run CELL 6 first: the frozen core is not loaded in this kernel')


def snapshot():
    csv = os.path.join(OUT, '{csv}')
    if not os.path.exists(csv):
        return
    z = os.path.join(OUT, '{snap}')
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv, '{csv}')
        for f in ('core_kaggle.py', '{driver}'):
            if os.path.exists(f):
                zf.write(f, f)
    print('  snapshot -> %s (%.0f KB)' % (z, os.path.getsize(z) / 1024))


LOG = boot()
print('rows already logged:', len(LOG.rows))

RUN_STATUS = {{}}
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
    print('  %-5s %s' % (k, RUN_STATUS.get(k, 'not reached')))
print('total %.2f h' % ((time.time() - t_all) / 3600))
'''

VGG_PREFLIGHT = r"""cleanup()

# VGG-16-BN memory preflight for E7: one full-size forward/backward before E1.
VGG_BS = 256
vm = net = None
try:
    vm = CIFARModel('vgg16_bn').to(dev)
    net = nn.DataParallel(vm) if NGPU > 1 else vm
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP):
        l = F.cross_entropy(net(augment_batch(Xtr[:VGG_BS])), ytr[:VGG_BS])
    l.backward()
    print('vgg     : forward/backward OK at batch %d (E7 will use this)' % VGG_BS)
except RuntimeError as e:
    if 'out of memory' not in str(e).lower():
        raise
    VGG_BS = 128
    print('vgg     : OOM at 256; E7 will use batch 128')
finally:
    try:
        del vm, net, l
    except Exception:
        pass
    cleanup()
"""


def intro_md(c):
    return f"""# FML &middot; {c['name']} &mdash; Kaggle run notebook (seed {c['seed_label']})

**What this is.** A self-contained notebook for the {c['name']} work package: {c['intro']}

**Requirements**
- Kaggle notebook, **Accelerator = GPU T4 x2**, **Internet = On** (or a CIFAR-10 dataset attached under Input).
- Expected wall-clock **{c['wall']}**: {c['times']}.

**How to run, in order**

| Cell | What it does | Time |
|---|---|---|
| 1 | environment check (GPU, disk, working dir) | 5 s |
| 2 | resume: restores `{c['csv']}` if an input dataset has one | 5 s |
| 3 | writes `core_kaggle.py` (frozen, do not edit) | 1 s |
| 4 | writes `{c['driver']}` (frozen, do not edit) | 1 s |
| 5 | verifies both files against the frozen hashes and compiles them | 2 s |
| 6 | loads the code, checks CIFAR shapes, one forward/backward smoke | 1-2 min |
| 7 | **runs all blocks** &mdash; the long cell | {c['wall']} |
| 8 | audit, gate summary, `{c['zip']}` | 20 s |

The manual numbers its cells differently because it has no separate environment or preflight cells: manual Cell 1 = this Cell 3, manual Cell 2 = this Cell 4, and manual Cell 3 = this Cells 5-7. When you work inside this file, follow the notebook's numbering.

**Resume.** Every measured row is flushed to `/kaggle/working/{c['csv']}` immediately. If the session dies, download that CSV, upload it to a private dataset named `fml-runs-{c['key']}`, attach the dataset, and run Cells 1 &rarr; 7 again. The resume key skips every completed row, so nothing is measured twice.

**{c['notes']}**

**Do not edit** `core_kaggle.py` or `{c['driver']}` inside this notebook. Cell 5 proves the text you ran is the text that was frozen.
"""


def send_md(c):
    return f"""## Send back

1. **`{c['zip']}`** &mdash; Output pane (right side) &rarr; `{c['zip']}` &rarr; download. It contains `{c['csv']}`, `GATE_SUMMARY.txt`, `gate_summary.json`, `meta.json`, the two frozen source files, and any pivot tables that apply.
2. **The `PASTE THIS TO THE OWNER` block** printed by Cell 8 &mdash; copy it into the message.

Do not hand-edit `{c['csv']}`. If a block failed, send `GATE_SUMMARY.txt` and the traceback from the log; the CSV keeps every row that was measured before the failure.

**If the session dies early:** download `/kaggle/working/{c['csv']}`, upload it to the private dataset `fml-runs-{c['key']}`, attach the dataset to a new session, and run Cells 1 &rarr; 7 again. Completed rows are skipped by the resume key, so only the unfinished part re-runs.
"""


def build_cell8(base8, c):
    m_e1 = "say('GATE 1 - E1 at recall 0.75, precision 1.0')"
    m_g2 = "say('GATE 2 - E9 amplification (3 seeds, 10% poison)')"
    m_tail = "\n\ndef find(p):"
    for m in (m_e1, m_g2, m_tail):
        assert base8.count(m) == 1, 'marker not unique: ' + m
    i1 = base8.index(m_e1)
    i2 = base8.index(m_g2)
    i3 = base8.index(m_tail)
    head, e1, tail = base8[:i1], base8[i1:i2], base8[i3:]

    body = head + (e1 if c['has_e1'] else '') + c['extra'] + tail

    reps = [
        ("say('FML / Sai (seed 0) - audit')",
         "say('FML / %s (seed %s) - audit')" % (c['name'], c['seed_label'])),
        ("CSV = os.path.join(W, 'runs_sai.csv')",
         "CSV = os.path.join(W, '%s')" % c['csv']),
        ("assert os.path.exists(CSV), 'runs_sai.csv not found - run CELL 7 first'",
         "assert os.path.exists(CSV), '%s not found - run CELL 7 first'" % c['csv']),
        ("EXPECT = {'E1': (53, 53), 'E9': (15, 15), 'E3': (21, 22), 'E6': (6, 6), 'E7b': (1, 1)}",
         'EXPECT = %r' % c['expect']),
        ("for e in ['E1', 'E9', 'E3', 'E6', 'E7b']:",
         'for e in %r:' % list(c['expect'].keys())),
        ("    person='sai', seed=0, notebook='fml-sai-main',",
         "    person='%s', seed=%d, notebook='%s'," % (c['key'], c['seed'], c['notebook'])),
        ("code_sha12={f: sha12(f) for f in ('core_kaggle.py', 'person_sai.py') if find(f)},",
         "code_sha12={f: sha12(f) for f in ('core_kaggle.py', '%s') if find(f)}," % c['driver']),
        ("csv='runs_sai.csv', rows=int(len(df)),",
         "csv='%s', rows=int(len(df))," % c['csv']),
        ("rows_per_exp={e: int((df.exp == e).sum()) for e in ['E1', 'E9', 'E3', 'E6', 'E7b']},",
         "rows_per_exp={e: int((df.exp == e).sum()) for e in %r}," % list(c['expect'].keys())),
        ("zpath = os.path.join(W, 'FML_Sai_results.zip')",
         "zpath = os.path.join(W, '%s')" % c['zip']),
        ("    zf.write(CSV, 'runs_sai.csv')",
         "    zf.write(CSV, '%s')" % c['csv']),
        ("    for f in ('core_kaggle.py', 'person_sai.py', 'E1_pivot_asr.csv', 'E1_pivot_acc.csv'):",
         "    for f in ('core_kaggle.py', '%s', 'E1_pivot_asr.csv', 'E1_pivot_acc.csv'):" % c['driver']),
        ("print('\\nDownload it from the right-hand Output pane: FML_Sai_results.zip')",
         "print('\\nDownload it from the right-hand Output pane: %s')" % c['zip']),
    ]
    for old, new in reps:
        assert body.count(old) == 1, 'replacement not unique: ' + old[:60]
        body = body.replace(old, new)
    return body


def build(c, base):
    drv_path = os.path.join(HERE, c['driver'])
    drv_txt = read_lf(drv_path)
    assert sha12(drv_txt) == c['sha'], (drv_path, sha12(drv_txt))
    assert drv_txt.count("if __name__ == '__main__':") == 1

    B = [''.join(x['source']) for x in base['cells']]

    # cell 0
    c0 = intro_md(c)
    # cell 1 unchanged
    c1 = B[1]
    # cell 2: resume, CSV name only
    c2 = B[2].replace('runs_sai.csv', c['csv'])
    assert 'runs_sai' not in c2
    # cell 3: core unchanged
    c3 = B[3]
    assert c3 == '%%writefile core_kaggle.py\n' + read_lf(CORE)
    # cell 4: driver payload
    c4 = '%%writefile ' + c['driver'] + '\n' + drv_txt
    # cell 5: hash check
    c5 = B[5]
    old5 = "EXPECT = {'core_kaggle.py': 'ad7a363926a5', 'person_sai.py': '45295331c333'}"
    assert c5.count(old5) == 1
    c5 = c5.replace(old5, "EXPECT = {'core_kaggle.py': '%s', '%s': '%s'}"
                   % (CORE_SHA, c['driver'], c['sha']))
    # cell 6: loader
    c6 = B[6].replace('person_sai.py', c['driver'])
    old6 = "setup('sai', seed=0)"
    assert c6.count(old6) == 1
    c6 = c6.replace(old6, "setup('%s', seed=%d)" % (c['key'], c['seed']))
    if c['vgg']:
        old6b = "print('READY')"
        assert c6.count(old6b) == 1
        c6 = c6.replace(old6b, VGG_PREFLIGHT + "\nprint('READY')")
    # cell 7: runner
    c7 = CELL7_TMPL.format(wall=c['wall'], blocks=c['blocks_js'], max_hours=c['max_hours'],
                           pre=c['pre7'], calls=c['calls'], csv=c['csv'],
                           snap=c['snap'], driver=c['driver'])
    # cell 8: audit
    c8 = build_cell8(B[8], c)
    # cell 9
    c9 = send_md(c)

    cells = [md(c0), code(c1), code(c2), code(c3), code(c4), code(c5),
             code(c6), code(c7), code(c8), md(c9)]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate(nb, c, out):
    raw = open(out, encoding='utf-8').read()
    nb2 = json.loads(raw)
    assert nb2['nbformat'] == 4 and len(nb2['cells']) == 10
    texts = [''.join(x['source']) for x in nb2['cells']]
    # writefile round-trip
    assert texts[3] == '%%writefile core_kaggle.py\n' + read_lf(CORE)
    assert texts[4] == '%%writefile ' + c['driver'] + '\n' + read_lf(os.path.join(HERE, c['driver']))
    # compile everything
    for i, x in enumerate(nb2['cells']):
        if x['cell_type'] != 'code':
            continue
        t = ''.join(x['source'])
        if t.startswith('%%writefile'):
            compile(t.split('\n', 1)[1], 'cell%d' % i, 'exec')
        else:
            compile(t, 'cell%d' % i, 'exec')
    # no Sai leakage
    for bad in ('runs_sai', 'person_sai', 'FML_Sai', "setup('sai'", 'fml-sai'):
        for i, t in enumerate(texts):
            assert bad not in t, 'leak %r in cell %d' % (bad, i)
    print('OK  %s  (%d bytes)' % (out, len(raw.encode('utf-8'))))


if __name__ == '__main__':
    base = json.load(open(BASE_NB, encoding='utf-8'))
    assert len(base['cells']) == 10
    for key in ('sarang', 'divyansh', 'urja'):
        c = CONFIGS[key]
        nb = build(c, base)
        out = os.path.join(OUTDIR, 'FML_%s_Kaggle_Run.ipynb' % c['name'])
        os.makedirs(OUTDIR, exist_ok=True)
        with open(out, 'w', encoding='utf-8', newline='\n') as fh:
            json.dump(nb, fh, ensure_ascii=False, indent=1)
            fh.write('\n')
        validate(nb, c, out)
