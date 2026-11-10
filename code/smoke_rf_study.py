"""Laptop smoke test for FML_RF_Removal_Study.ipynb.

Builds a format-faithful MINIATURE RML2016.10a_dict.pkl (11 mods x 3 SNRs x
60 samples, shape (60,2,128), pickle protocol 2, str/int tuple keys) so the
real loader/pipeline is exercised without the 641 MB download. Runs the
notebook end-to-end with a reduced grid (1 seed, 2 epochs, 2 cells, all 4
removers), checks CSV/per-SNR/gamma*/figures/zip, then RE-RUNS the long cell
to confirm resume skips finished rows without duplicates. Deletes the scratch
directory afterward. Summary -> E:\\FML\\team\\smoke_rf_study_summary.json.
"""
import json
import os
import pickle
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

NB = Path(r'E:\FML\handouts\notebooks\FML_RF_Removal_Study.ipynb')
SCRATCH = Path(r'E:\FML\_smoke_rf_study')
SUMMARY = Path(r'E:\FML\team\smoke_rf_study_summary.json')
MODS = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64', 'GFSK', 'CPFSK',
        'AM-DSB', 'AM-SSB', 'WBFM']
SNRS = [0, 10, 18]


def src(cell):
    return ''.join(cell['source'])


def make_miniature(path):
    rng = np.random.RandomState(0)
    d = {}
    for m in MODS:
        for s in SNRS:
            d[(m, s)] = rng.randn(60, 2, 128).astype(np.float32)
    with open(path, 'wb') as f:
        pickle.dump(d, f, protocol=2)


def main():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    make_miniature(SCRATCH / 'RML2016.10a_dict.pkl')
    old = Path.cwd()
    os.chdir(SCRATCH)
    result = {}
    try:
        nb = json.loads(NB.read_text(encoding='utf-8'))
        ns = {'__name__': '__main__'}
        run_all_src = None
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            s = src(cell)
            cid = cell['id']
            if s.lstrip().startswith('# CELL 4'):
                run_all_src = s
            exec(compile(s, '<RFstudy:%s>' % cid, 'exec'), ns)
            if 'CELL 1' in s[:40]:
                ns.update(SEEDS=[0], EPOCHS=2, BS=64, FT_EPOCHS=1,
                          SNR_SET=[0, 10, 18], CELLS=[(1.0, 1.0), (0.5, 1.0)])
                print('  [%s] miniature + overrides (SEEDS=[0],EPOCHS=2,BS=64,FT=1,2 cells)' % cid, flush=True)
            else:
                print('  [%s] ok' % cid, flush=True)

        import pandas as pd
        rows = int(len(pd.read_csv('runs_rf_study.csv')))
        per = int(len(pd.read_csv('rf_study_per_snr.csv')))
        gamma = Path('gamma_star.csv').exists()
        fig1 = Path('rf_study_asr_snr.png').exists()
        fig2 = Path('rf_study_gamma_snr.png').exists()
        zipf = Path('FML_RF_Study.zip').exists()
        expected = 1 * (1 + 2 * 4)   # 1 seed x (1 base + 2 cells x 4 methods) = 9

        # resume test: re-run the long cell; finished seed must be skipped, no dups
        assert run_all_src is not None, 'run_all cell not found'
        exec(compile(run_all_src, '<RFstudy:resume>', 'exec'), ns)
        df2 = pd.read_csv('runs_rf_study.csv')
        rows_after = int(len(df2))
        dup = int(df2.duplicated(subset=['seed', 'method', 'recall', 'precision']).sum())
        resume_ok = (rows_after == rows) and dup == 0

        ok = (rows == expected and per > 0 and gamma and fig1 and fig2
              and zipf and resume_ok)
        result = dict(status='PASS' if ok else 'FAIL', csv_rows=rows,
                      expected=expected, per_snr_rows=per, gamma_star=gamma,
                      figure_asr=fig1, figure_gamma=fig2, zip=zipf,
                      resume_rows_after=rows_after, resume_dup=dup, resume_ok=resume_ok)
        print('RESULT', result, flush=True)
    except Exception as e:
        result = dict(status='FAIL', error=str(e), trace=traceback.format_exc())
        print('EXCEPTION\n' + traceback.format_exc(), flush=True)
    finally:
        os.chdir(old)
        SUMMARY.write_text(json.dumps(result, indent=2), encoding='utf-8')
        shutil.rmtree(SCRATCH, ignore_errors=True)
        print('scratch deleted:', not SCRATCH.exists(), flush=True)
    return 0 if result.get('status') == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())