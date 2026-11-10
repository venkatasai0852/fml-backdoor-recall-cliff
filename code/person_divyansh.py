"""
DRIVER C -- DIVYANSH (seed 2). Kaggle: Cell 1 = core_kaggle.py, Cell 2 = this file.
Blocks: E1 (main grid, seed 2) -> E7-B (Blend on ResNet-18 + trigger synthesis)
        -> SISA baseline (K=5 shards, r in {1.0, 0.5}).
Set MAIN below. Everything resumes from /kaggle/working/runs_divyansh.csv.
"""
import os, time
try:
    setup
except NameError:
    from core_kaggle import *

NAME = 'divyansh'
ARCH, ATTACK, RATE = 'resnet18', 'badnets', 0.05


def boot():
    setup(NAME, seed=2)
    load_cifar()
    return Logger(f'{OUT}/runs_{NAME}.csv')


# ---------------------------------------------------------------- E1
def run_e1(LOG, seed=2):
    arch, attack, rate = ARCH, ATTACK, RATE
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    n_poison = int(ip.sum())
    for det in ['ss', 'ac']:
        if not LOG.has(exp='E1', arch=arch, attack=attack, rate=rate, seed=seed,
                       recall=1.0, precision=1.0, n_forget=n_poison,
                       method='poisoned', detector=det):
            log_detection(LOG, 'E1', arch, attack, rate, base, Xp, yp, ip,
                          detector=det, recall=1.0, precision=1.0, n_forget=n_poison)
    if not LOG.has(exp='E1', arch=arch, attack=attack, rate=rate, seed=seed,
                   recall=1.0, precision=1.0, n_forget=n_poison,
                   method='nc_style', detector='none'):
        t0 = time.time()
        m = inoculated_finetune(base, Xp, yp, attack)
        log_removal(LOG, m, 'E1', arch, attack, rate, 'nc_style', 1.0, 1.0,
                    n_poison, time.time() - t0, notes='inoculation only; F-independent')
        del m; cleanup()
    for (r, p) in E1_CONFIGS:
        F, st = make_forget_set(ip, yp, r, p, seed)
        for meth in E1_METHODS:
            if LOG.has(exp='E1', arch=arch, attack=attack, rate=rate, seed=seed,
                       recall=st['recall'], precision=st['precision'],
                       n_forget=st['n_forget'], method=meth, detector='none'):
                continue
            m, t = apply_method(meth, base, Xp, yp, F, attack)
            log_removal(LOG, m, 'E1', arch, attack, rate, meth,
                        st['recall'], st['precision'], st['n_forget'], t)
            del m; cleanup()
    del base; cleanup()


# ---------------------------------------------------------------- E7-B
def run_e7b(LOG, seed=2):
    arch, attack, rate = 'resnet18', 'blend', 0.05
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    n_poison = int(ip.sum())
    Ftrue = ip.cpu().numpy()
    key = dict(exp='E7', arch=arch, attack=attack, rate=rate, seed=seed,
               recall=1.0, precision=1.0, n_forget=n_poison, method='gold',
               detector='none')
    if not LOG.has(**key):
        m, t = apply_method('retrain_delF', base, Xp, yp, Ftrue, attack)
        log_removal(LOG, m, 'E7', arch, attack, rate, 'gold', 1.0, 1.0,
                    n_poison, t, notes='oracle retrain')
        del m; cleanup()
    for r in [0.75, 0.50]:
        F, st = make_forget_set(ip, yp, r, 1.0, seed)
        for meth in ['retrain_delF', 'recipe']:
            key = dict(exp='E7', arch=arch, attack=attack, rate=rate, seed=seed,
                       recall=st['recall'], precision=st['precision'],
                       n_forget=st['n_forget'], method=meth, detector='none')
            if LOG.has(**key):
                continue
            m, t = apply_method(meth, base, Xp, yp, F, attack)
            log_removal(LOG, m, 'E7', arch, attack, rate, meth,
                        st['recall'], st['precision'], st['n_forget'], t,
                        notes='Blend cell')
            del m; cleanup()
    # trigger synthesis on Blend, then inoculation with the synthesized trigger
    ck = f'{OUT}/synth_{arch}_{attack}_s{seed}.pt'
    if os.path.exists(ck):
        d = torch.load(ck, map_location='cpu')
        mask, pat = d['mask'], d['pattern']
    else:
        b = synthesize_trigger(base, Xp, yp)
        mask, pat = b['mask'], b['pattern']
        torch.save(dict(mask=mask, pattern=pat, src_class=b['src_class']), ck)
    key = dict(exp='E7b', arch=arch, attack=attack, rate=rate, seed=seed,
               recall=1.0, precision=1.0, n_forget=n_poison,
               method='nc_synth', detector='none')
    if not LOG.has(**key):
        t0 = time.time()
        m = inoculated_finetune(base, Xp, yp, attack,
                                stamp_fn=lambda z: apply_synth_trigger(z, mask, pat))
        log_removal(LOG, m, 'E7b', arch, attack, rate, 'nc_synth', 1.0, 1.0,
                    n_poison, time.time() - t0,
                    notes=f'synth trigger on Blend; mask_mass={float(mask.sum()):.0f}')
        del m; cleanup()
    del base, Xp; cleanup()


# ---------------------------------------------------------------- SISA
def _sisa_train(Xp, yp, F, K=5, epochs=12, tag='sisa'):
    """Train K shard models on poisoned shards; retrain a shard only if it holds flags."""
    n = len(yp)
    shards = np.array_split(np.random.RandomState(SEED).permutation(n), K)
    models = []
    for j, ids in enumerate(shards):
        m = CIFARModel('resnet18').to(dev)
        m, _ = train_model(Xp[ids], yp[ids], model=m, epochs=epochs, tag=f'{tag}_base{j}')
        if F[ids].any():
            keep = ids[~F[ids]]
            m = CIFARModel('resnet18').to(dev)
            m, _ = train_model(Xp[keep], yp[keep], model=m, epochs=epochs,
                               tag=f'{tag}_retrain{j}')
        m.cpu().eval()
        models.append(m)
        cleanup()
    return models


def _eval_ensemble(models, attack):
    dev_models = [m.to(dev) for m in models]
    prob = sum(predict_probs(m, Xte) for m in dev_models) / len(dev_models)
    acc = (prob.argmax(1) == yte).float().mean().item() * 100
    nt = (yte != TARGET)
    xt = add_trigger(Xte[nt], attack)
    pt = sum(predict_probs(m, xt) for m in dev_models) / len(dev_models)
    asr = (pt.argmax(1) == TARGET).float().mean().item() * 100
    return round(acc, 2), round(asr, 2)


def run_sisa(LOG, seed=2):
    arch, attack, rate = ARCH, ATTACK, RATE
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    n_poison = int(ip.sum())
    for r in [1.0, 0.5]:
        F, st = make_forget_set(ip, yp, r, 1.0, seed)
        key = dict(exp='SISA', arch=arch, attack=attack, rate=rate, seed=seed,
                   recall=st['recall'], precision=st['precision'],
                   n_forget=st['n_forget'], method='sisa5', detector='none')
        if LOG.has(**key):
            continue
        t0 = time.time()
        models = _sisa_train(Xp, yp, F, K=5)
        acc, asr = _eval_ensemble(models, attack)
        LOG.log(**key, clean_acc=acc, asr=asr, time_sec=time.time() - t0,
                notes='K=5 shards, retrain shards containing flags; majority vote')
        del models; cleanup()
    del base, Xp; cleanup()


MAIN = ['E1', 'E7B', 'SISA']

if __name__ == '__main__':
    LOG = boot()
    for blk in MAIN:
        print(f'\n===== {blk} =====')
        t0 = time.time()
        {'E1': lambda: run_e1(LOG, 2), 'E7B': lambda: run_e7b(LOG, 2),
         'SISA': lambda: run_sisa(LOG, 2)}[blk]()
        print(f'===== {blk} done in {(time.time()-t0)/60:.1f} min =====')
    print(f'ALL DONE -> {LOG.path}')
