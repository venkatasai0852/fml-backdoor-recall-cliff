"""
DRIVER B -- SARANG (seed 1). Kaggle: Cell 1 = core_kaggle.py, Cell 2 = this file.
Blocks: E1 (main grid, seed 1) -> E7 (VGG-16-BN x BadNets/Blend robustness)
        -> E8 (ascent-strength cliff: inert vs collapse vs label repair).
Set MAIN below. Everything resumes from /kaggle/working/runs_sarang.csv.
"""
import os, time
try:
    setup
except NameError:
    from core_kaggle import *

NAME = 'sarang'
ARCH, ATTACK, RATE = 'resnet18', 'badnets', 0.05


def boot():
    setup(NAME, seed=1)
    load_cifar()
    return Logger(f'{OUT}/runs_{NAME}.csv')


# ---------------------------------------------------------------- E1
def run_e1(LOG, seed=1):
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


# ---------------------------------------------------------------- E7 (VGG)
def run_e7(LOG, seed=1, attacks=('badnets', 'blend')):
    arch, rate = 'vgg16_bn', 0.05
    for attack in attacks:
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
                            st['recall'], st['precision'], st['n_forget'], t)
                del m; cleanup()
        del base, Xp; cleanup()


# ---------------------------------------------------------------- E8
def _ascent(model0, X, y, idx, lr, epochs, bs=512):
    m = clone_model(model0)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    Xf = X[idx]; yf = y[idx]
    for ep in range(epochs):
        for k in range(0, len(idx), bs):
            xb = Xf[k:k + bs]; yb = yf[k:k + bs]
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(set_to_none=True)
            (-loss).backward()
            opt.step()
    m.eval()
    return m


def _feat_norm(model, X, n=2000):
    with torch.no_grad():
        f = extract_features(model, X[:n])
    return float(f.norm(dim=1).mean())


def run_e8(LOG, seed=1):
    arch, attack, rate = 'resnet18', 'badnets', 0.05
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    ipn = ip.cpu().numpy()
    pidx = np.where(ipn)[0]
    flagged = np.random.RandomState(11).permutation(pidx)[:len(pidx) // 2]
    F = np.zeros(len(yp), bool); F[flagged] = True
    for lr, ep in [(2e-4, 2), (5e-4, 2), (1e-3, 2), (1e-3, 4)]:
        tag = f'ascent_lr{lr}_ep{ep}'
        key = dict(exp='E8', arch=arch, attack=attack, rate=rate, seed=seed,
                   recall=0.5, precision=1.0, n_forget=len(flagged),
                   method=tag, detector='none')
        if LOG.has(**key):
            continue
        m = _ascent(base, Xp, yp, flagged, lr, ep)
        acc, asr = evaluate(m, attack)
        LOG.log(**key, clean_acc=acc, asr=asr, time_sec=0,
                notes=f'feature_norm={_feat_norm(m, Xp):.1f}')
        del m; cleanup()
    for lr, ep in [(0.005, 2), (0.01, 1)]:
        tag = f'repair_lr{lr}_ep{ep}'
        key = dict(exp='E8', arch=arch, attack=attack, rate=rate, seed=seed,
                   recall=0.5, precision=1.0, n_forget=len(flagged),
                   method=tag, detector='none')
        if LOG.has(**key):
            continue
        m = recipe_finetune(base, Xp, yp, F, attack, epochs=ep, lr=lr,
                            innoc=False, decouple=False)
        acc, asr = evaluate(m, attack)
        LOG.log(**key, clean_acc=acc, asr=asr, time_sec=0,
                notes=f'label-repair alternative; feature_norm={_feat_norm(m, Xp):.1f}')
        del m; cleanup()
    del base, Xp; cleanup()


MAIN = ['E1', 'E7', 'E8']

if __name__ == '__main__':
    LOG = boot()
    for blk in MAIN:
        print(f'\n===== {blk} =====')
        t0 = time.time()
        {'E1': lambda: run_e1(LOG, 1), 'E7': lambda: run_e7(LOG, 1),
         'E8': lambda: run_e8(LOG, 1)}[blk]()
        print(f'===== {blk} done in {(time.time()-t0)/60:.1f} min =====')
    print(f'ALL DONE -> {LOG.path}')
