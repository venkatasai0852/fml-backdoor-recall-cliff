"""
DRIVER D -- URJA (seeds 0,1,2). Kaggle: Cell 1 = core_kaggle.py, Cell 2 = this file.
Blocks: E2 (residual-poison cliff, 3 seeds) -> E4 (recipe ablations, seed 0)
        -> E5 (relearning + MIA, seed 0).
Set MAIN below. Everything resumes from /kaggle/working/runs_urja.csv.
NOTE: E2 alone is ~5 GPU-h. Split into two Kaggle sessions (E2 seeds 0-1, then seed 2 + E4/E5).
"""
import os, time
try:
    setup
except NameError:
    from core_kaggle import *

NAME = 'urja'
ARCH, ATTACK, RATE = 'resnet18', 'badnets', 0.05


def boot():
    setup(NAME, seed=0)
    load_cifar()
    return Logger(f'{OUT}/runs_{NAME}.csv')


# ---------------------------------------------------------------- E2
def run_e2(LOG, seeds=(0, 1, 2)):
    arch, attack, rate = ARCH, ATTACK, RATE
    for seed in seeds:
        base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
        n_poison = int(ip.sum())
        for n_res in E2_RESIDUAL:
            r = (n_poison - n_res) / n_poison
            F, st = make_forget_set(ip, yp, r, 1.0, seed)
            for meth in ['retrain_delF', 'recipe']:
                key = dict(exp='E2', arch=arch, attack=attack, rate=rate, seed=seed,
                           recall=st['recall'], precision=st['precision'],
                           n_forget=st['n_forget'], method=meth, detector='none')
                if LOG.has(**key):
                    continue
                if meth == 'recipe' and F.sum() == 0:
                    t0 = time.time()
                    m = inoculated_finetune(base, Xp, yp, attack)
                    t = time.time() - t0
                else:
                    m, t = apply_method(meth, base, Xp, yp, F, attack)
                log_removal(LOG, m, 'E2', arch, attack, rate, meth,
                            st['recall'], st['precision'], st['n_forget'], t,
                            notes=f'residual={n_res}')
                del m; cleanup()
        del base, Xp; cleanup()


# ---------------------------------------------------------------- E4
def run_e4(LOG, seed=0):
    arch, attack, rate = ARCH, ATTACK, RATE
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    for (r, p) in [(0.90, 1.0), (0.50, 1.0), (0.90, 0.70), (0.50, 0.70)]:
        F, st = make_forget_set(ip, yp, r, p, seed)
        cells = dict(exp='E4', arch=arch, attack=attack, rate=rate, seed=seed,
                     recall=st['recall'], precision=st['precision'],
                     n_forget=st['n_forget'], detector='none')
        # --- label-source diagnostics ---
        ref = None
        lab = {}
        for src in ['penult', 'early', 'refmodel']:
            key = dict(cells, method='label_acc')
            if src == 'refmodel':
                key['detector'] = 'refmodel'
                if LOG.has(**key):
                    continue
                pseudo, ref = refmodel_labels(base, Xp, yp, F, epochs=6)
            else:
                key['detector'] = src
                if LOG.has(**key):
                    continue
                pseudo = knn_repair(base, Xp, yp, F, layer=src)
            lab[src] = float(pseudo_accuracy(pseudo, yp, F, ip))
            LOG.log(**key, notes=f'label_acc={lab[src]:.4f}')
        if ref is None:
            ref = refmodel_labels(base, Xp, yp, F, epochs=6)[1]
        # --- variants ---
        variants = [
            ('recipe', dict()),
            ('recipe_decouple', dict(decouple=True)),
            ('recipe_noinnoc', dict(innoc=False)),
            ('recipe_early', dict(label_source='early')),
            ('recipe_refmodel', dict(label_source='refmodel', ref_model=ref)),
        ]
        for tag, kw in variants:
            key = dict(cells, method=tag)
            if LOG.has(**key):
                continue
            m = recipe_finetune(base, Xp, yp, F, attack, **kw)
            acc, asr = evaluate(m, attack)
            LOG.log(**key, clean_acc=acc, asr=asr, time_sec=0,
                    notes=f'label_acc penult={lab.get("penult", float("nan")):.3f} '
                          f'early={lab.get("early", float("nan")):.3f} '
                          f'ref={lab.get("refmodel", float("nan")):.3f}')
            del m; cleanup()
        del ref; cleanup()
    del base, Xp; cleanup()


# ---------------------------------------------------------------- E5
def run_e5(LOG, seed=0):
    arch, attack, rate = ARCH, ATTACK, RATE
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    F, st = make_forget_set(ip, yp, 0.90, 1.0, seed)
    m_recipe = recipe_finetune(base, Xp, yp, F, attack)
    m_gold, _ = apply_method('retrain_delF', base, Xp, yp, ip.cpu().numpy(), attack)
    for tag, m in [('poisoned', base), ('recipe', m_recipe), ('gold', m_gold)]:
        key = dict(exp='E5', arch=arch, attack=attack, rate=rate, seed=seed,
                   recall=st['recall'], precision=st['precision'],
                   n_forget=st['n_forget'], method='relearn', detector=tag)
        if not LOG.has(**key):
            steps = relearn_steps(m, Xtr, ytr, attack, n=100, max_steps=200)
            LOG.log(**key, notes=f'steps_to_asr90={steps} (max 200)')
        key = dict(key, method='mia')
        if not LOG.has(**key):
            a = mia_auc(m, Xtr, ytr, Xte, yte)
            LOG.log(**key, auc=a, notes='loss-threshold MIA, member=train subset')
        acc, asr = evaluate(m, attack)
        print(f'  [E5] {tag}: acc={acc} asr={asr}')
    del m_recipe, m_gold, base, Xp; cleanup()


MAIN = ['E2', 'E4', 'E5']

if __name__ == '__main__':
    LOG = boot()
    for blk in MAIN:
        print(f'\n===== {blk} =====')
        t0 = time.time()
        {'E2': lambda: run_e2(LOG), 'E4': lambda: run_e4(LOG, 0),
         'E5': lambda: run_e5(LOG, 0)}[blk]()
        print(f'===== {blk} done in {(time.time()-t0)/60:.1f} min =====')
    print(f'ALL DONE -> {LOG.path}')
