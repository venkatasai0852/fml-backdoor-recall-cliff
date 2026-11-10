"""Shared helpers for the A-G Kaggle notebook generators.

Every notebook embeds the frozen core byte-for-byte (LF, sha256-pinned),
hash-checks it, execs it into the notebook globals, then layers new
experiment functions on top. tqdm on every driver loop; the last cell
audits the CSV, zips it with a meta JSON, and shows a direct download link.
"""
import hashlib
import json
from pathlib import Path

TEAM = Path(r'E:\FML\team')
NB_DIR = Path(r'E:\FML\handouts\notebooks')
CORE_PATH = TEAM / 'core_kaggle.py'
CORE = CORE_PATH.read_bytes().replace(b'\r\n', b'\n').decode('utf-8')
CORE_SHA = hashlib.sha256(CORE.encode('utf-8')).hexdigest()
SRC_NB = NB_DIR / 'FML_Sarang_Kaggle_Run.ipynb'
META = json.loads(SRC_NB.read_text(encoding='utf-8'))['metadata']


class NB:
    def __init__(self):
        self.cells = []

    def md(self, text):
        self.cells.append({'cell_type': 'markdown', 'id': 'c%d' % len(self.cells),
                           'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)})

    def code(self, text):
        self.cells.append({'cell_type': 'code', 'id': 'c%d' % len(self.cells),
                           'execution_count': None, 'metadata': {}, 'outputs': [],
                           'source': text.strip('\n').splitlines(keepends=True)})

    def code_exact(self, text):
        self.cells.append({'cell_type': 'code', 'id': 'c%d' % len(self.cells),
                           'execution_count': None, 'metadata': {}, 'outputs': [],
                           'source': text.splitlines(keepends=True)})

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nb = {'cells': self.cells, 'metadata': META, 'nbformat': 4, 'nbformat_minor': 5}
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print('wrote', path, path.stat().st_size, 'bytes |', len(self.cells), 'cells')


ENV_CHECK = r"""
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
print('tqdm    ', __import__('tqdm').__version__)
"""


def head_cells(nb, knobs_code):
    """Cell 1 (knobs+env), Cell 2 (writefile core), Cell 3 (hash), Cell 4 (exec+data)."""
    nb.code('# CELL 1 - knobs and environment check. Nothing heavy happens here.\n'
            'import os, sys, json, time, platform, hashlib, zipfile, shutil, math\n'
            'from tqdm.auto import tqdm\n'
            + knobs_code.strip('\n') + '\n' + ENV_CHECK)
    nb.code_exact('%%writefile core_kaggle.py\n' + CORE)
    nb.code(r"""
# CELL 3 - the frozen core must arrive byte-identical
EXPECT = '__SHA__'
raw = open('core_kaggle.py', 'rb').read()
got = hashlib.sha256(raw).hexdigest()
assert got == EXPECT, 'core_kaggle.py hash mismatch: %s != %s' % (got, EXPECT)
print('core_kaggle.py  %d bytes  sha256[:12]=%s  OK' % (len(raw), got[:12]))
""".replace('__SHA__', CORE_SHA))
    nb.code(r"""
# CELL 4 - load the frozen core into this notebook, then CIFAR-10
src = open('core_kaggle.py').read()
exec(compile(src, 'core_kaggle.py', 'exec'))
print('device', dev, '| gpus', NGPU, '| BS', BS, '| epochs', EPOCHS)
load_cifar()
""")


def final_cell(nb, zipname, csvname, purpose, expected_expr, meta_extra=''):
    """Audit the CSV, write meta JSON, zip, and show a direct download link."""
    nb.code(r"""
# LAST CELL - audit, package, download link
import pandas as pd
from IPython.display import FileLink, display

PATH = '__CSV__'
ZIP = '__ZIP__'

def audit(path):
    if not os.path.exists(path):
        print('MISSING', path, '- run the long cell first')
        return False, 0
    d = pd.read_csv(path)
    key_cols = ['exp', 'arch', 'attack', 'rate', 'seed', 'recall',
                'precision', 'n_forget', 'method', 'detector']
    dup = int(d.duplicated(subset=key_cols).sum())
    print('\n--- %s ---' % path)
    print('rows %d | columns %d | duplicate keys %d' % (len(d), d.shape[1], dup))
    for exp in sorted(d.exp.unique()):
        print('  %-6s %d rows' % (exp, int((d.exp == exp).sum())))
    bad = d[d.clean_acc.isna() | d.asr.isna()]
    print('missing acc/asr %d | time_sec<=0 %d' % (len(bad), int((d.time_sec <= 0).sum())))
    expected = __EXPECTED__
    ok = (len(d) >= expected) and dup == 0
    print('expected >= %d rows -> %s' % (expected, 'PASS' if ok else 'CHECK ABOVE'))
    return ok, len(d)

ok, nrows = audit(PATH)

meta = dict(
    purpose='__PURPOSE__',
    core_sha256=EXPECT,
    rows_expected=__EXPECTED__,
    rows_actual=nrows,
    audit_pass=bool(ok),
    generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    host=platform.platform(),
    __EXTRA__
)
json.dump(meta, open('__METAJSON__', 'w'), indent=2)

with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in [PATH, '__METAJSON__']:
        if os.path.exists(f):
            z.write(f)

print('\n============================================================')
print('DOWNLOAD')
print('============================================================')
print('zip: /kaggle/working/%s  (%d rows)' % (ZIP, nrows))
print('Click the link below, or use the Output tab (right panel) ->')
print('  /kaggle/working/%s -> three dots -> Download.' % ZIP)
print('============================================================')
display(FileLink(ZIP))
""".replace('__CSV__', csvname)
   .replace('__ZIP__', zipname)
   .replace('__PURPOSE__', purpose)
   .replace('__EXPECTED__', expected_expr)
   .replace('__METAJSON__', zipname.replace('.zip', '_meta.json'))
   .replace('__EXTRA__', meta_extra))