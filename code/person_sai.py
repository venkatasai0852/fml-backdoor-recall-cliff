"""
DRIVER A -- SAI (seed 0). Kaggle: Cell 1 = core_kaggle.py, Cell 2 = this file.
Blocks: E1 (main grid, seed 0) -> E9 (amplification at scale, 3 seeds)
        -> E3 (Calibrate-then-Remove) -> E6 (realistic detector column)
        -> E7b (NC trigger synthesis + inoculation with synthesized trigger).
Set MAIN below. Everything resumes from /kaggle/working/runs_sai.csv.
"""
import os, time
try:
    setup
except NameError:
    from core_kaggle import *

NAME = 'sai'
ARCH, ATTACK, RATE = 'resnet18', 'badnets', 0.05


def boot():
    setup(NAME, seed=0)
    load_cifar()
    return Logger(f'{OUT}/runs_{NAME}.csv')


# ---------------------------------------------------------------- E1
def run_e1(LOG, seed=0):
    arch, attack, rate = ARCH, ATTACK, RATE
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    n_poison = int(ip.sum())
    ipn = ip.cpu().numpy()
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


# ---------------------------------------------------------------- E9
def _amp_log(LOG, model, tag, arch, attack, rate, seed, missed, flagged, Xp, yp, ip):
    ipn = ip.cpu().numpy()
    c0 = (yp == TARGET).cpu().numpy()
    neg = c0 & ~ipn
    sc = ss_scores(extract_features(model, Xp), yp)
    auc = auc_score(sc[missed], sc[neg])
    tpr = tpr_at_fpr(sc[missed], sc[neg])
    top = np.argsort(sc)[-len(flagged):]
    fl = np.zeros(len(sc), bool); fl[top] = True
    topk = float(fl[missed].mean())
    acc, asr = evaluate(model, attack)
    LOG.log(exp='E9', arch=arch, attack=attack, rate=rate, seed=seed,
            recall=0.5, precision=1.0, n_forget=len(flagged), method=tag,
            detector='ss', clean_acc=acc, asr=asr, auc=auc, tpr_at_fpr10=tpr,
            n_flags=int(fl.sum()), n_true_in_flags=int((fl & ipn).sum()),
            notes=f'AUC=missed-vs-clean0, topk_missed_recall={topk:.3f}')
    return auc


def run_e9(LOG, seeds=(0, 1, 2)):
    arch, attack, rate = 'resnet18', 'badnets', 0.10
    for seed in seeds:
        base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
        ipn = ip.cpu().numpy()
        pidx = np.where(ipn)[0]
        perm = np.random.RandomState(123).permutation(len(pidx))
        flagged = pidx[perm[:len(pidx) // 2]]
        missed = pidx[perm[len(pidx) // 2:]]
        F = np.zeros(len(yp), bool); F[flagged] = True
        if not LOG.has(exp='E9', arch=arch, attack=attack, rate=rate, seed=seed,
                       recall=0.5, precision=1.0, n_forget=len(flagged),
                       method='base', detector='ss'):
            _amp_log(LOG, base, 'base', arch, attack, rate, seed, missed, flagged, Xp, yp, ip)
        for lr, ep in [(0.005, 2), (0.01, 1)]:
            tag = f'partial_repair_lr{lr}_ep{ep}'
            if LOG.has(exp='E9', arch=arch, attack=attack, rate=rate, seed=seed,
                       recall=0.5, precision=1.0, n_forget=len(flagged),
                       method=tag, detector='ss'):
                continue
            m = recipe_finetune(base, Xp, yp, F, attack, epochs=ep, lr=lr,
                                innoc=False, decouple=False)
            _amp_log(LOG, m, tag, arch, attack, rate, seed, missed, flagged, Xp, yp, ip)
            del m; cleanup()
        rc = np.random.RandomState(7).choice(np.where(~ipn)[0], len(flagged), replace=False)
        Fc = np.zeros(len(yp), bool); Fc[rc] = True
        for lr, ep in [(0.005, 2), (0.01, 1)]:
            tag = f'ctrl_clean_lr{lr}_ep{ep}'
            if LOG.has(exp='E9', arch=arch, attack=attack, rate=rate, seed=seed,
                       recall=0.5, precision=1.0, n_forget=len(flagged),
                       method=tag, detector='ss'):
                continue
            m = recipe_finetune(base, Xp, yp, Fc, attack, epochs=ep, lr=lr,
                                innoc=False, decouple=False)
            _amp_log(LOG, m, tag, arch, attack, rate, seed, missed, flagged, Xp, yp, ip)
            del m; cleanup()
        del base, Xp; cleanup()


# ---------------------------------------------------------------- E3
def _synth_cached(base, Xp, yp, seed, tag):
    ck = f'{OUT}/synth_{tag}_s{seed}.pt'
    if os.path.exists(ck):
        d = torch.load(ck, map_location='cpu')
        return d['mask'], d['pattern']
    b = synthesize_trigger(base, Xp, yp)
    d = dict(mask=b['mask'], pattern=b['pattern'], src_class=b['src_class'])
    torch.save(d, ck)
    print(f'  synth saved {ck} (src_class={b["src_class"]}, norm={b["norm"]:.1f})')
    return d['mask'], d['pattern']


def _tau(model, mask, pattern, n=1000):
    nt = (yte != TARGET).nonzero(as_tuple=False).squeeze(1)[:n]
    xt = apply_synth_trigger(Xte[nt], mask, pattern)
    return round(float((predict_labels(model, xt) == TARGET).float().mean()) * 100, 2)


def run_e3(LOG, seed=0):
    arch, attack, rate = ARCH, ATTACK, RATE
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    ipn = ip.cpu().numpy()
    mask, pat = _synth_cached(base, Xp, yp, seed, f'{arch}_{attack}')
    true_m = np.zeros((32, 32), bool); true_m[27:31, 27:31] = True
    sm = (mask[0, 0].numpy() > 0.5)
    iou = float((sm & true_m).sum() / max((sm | true_m).sum(), 1))
    c0 = (yp == TARGET).cpu().numpy()
    sc = ss_scores(extract_features(base, Xp), yp)
    ths = sorted(set(np.quantile(sc[c0], np.linspace(0.0, 0.999, 20))))
    rows = []
    for th in ths:
        F = c0 & (sc >= th)
        tp = int((F & ipn).sum())
        prec = tp / max(int(F.sum()), 1); rec = tp / max(int(ipn.sum()), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        key = dict(exp='E3', arch=arch, attack=attack, rate=rate, seed=seed,
                   recall=round(rec, 4), precision=round(prec, 4),
                   n_forget=int(F.sum()), method='car_proxy', detector='ss')
        if LOG.has(**key):
            continue
        t0 = time.time()
        proxy = finetune_retain(base, Xp, yp, F, epochs=2, lr=0.01)
        tv = _tau(proxy, mask, pat)
        acc, asr = evaluate(proxy, attack)
        LOG.log(**key, clean_acc=acc, asr=asr,
                notes=f'theta={th:.4f} tau={tv} f1={f1:.3f} iou={iou:.3f} t={time.time()-t0:.0f}s')
        rows.append(dict(th=th, F=F, prec=prec, rec=rec, f1=f1, tau=tv))
        del proxy; cleanup()
    # rebuild rows from CSV if resumed (proxy rows already logged)
    if not rows:
        df = pd.read_csv(LOG.path)
        df = df[(df.exp == 'E3') & (df.method == 'car_proxy') & (df.seed == seed)]
        for _, r in df.iterrows():
            rows.append(dict(th=float(str(r.notes).split('theta=')[1].split()[0]),
                             F=c0 & (sc >= float(str(r.notes).split('theta=')[1].split()[0])),
                             prec=r.precision, rec=r.recall, f1=float(str(r.notes).split('f1=')[1].split()[0]),
                             tau=float(str(r.notes).split('tau=')[1].split()[0])))
    if not rows:
        print('E3: no proxy rows'); return
    best_f1 = max(rows, key=lambda d: d['f1'])
    feas = [d for d in rows if d['tau'] <= 5.0]
    car = max(feas, key=lambda d: d['th']) if feas else None
    print(f'E3: theta_F1={best_f1["th"]:.4f} f1={best_f1["f1"]:.3f} tau={best_f1["tau"]} '
          f'| CaR={"none" if car is None else round(car["th"],4)} iou={iou:.3f}')
    for tag, cfg in [('f1_full', best_f1), ('car_full', car)]:
        if cfg is None:
            continue
        key = dict(exp='E3', arch=arch, attack=attack, rate=rate, seed=seed,
                   recall=round(cfg['rec'], 4), precision=round(cfg['prec'], 4),
                   n_forget=int(cfg['F'].sum()), method=tag, detector='ss')
        if LOG.has(**key):
            continue
        m, t = apply_method('retrain_delF', base, Xp, yp, cfg['F'], attack)
        acc, asr = evaluate(m, attack)
        LOG.log(**key, clean_acc=acc, asr=asr, time_sec=t,
                notes=f'theta={cfg["th"]:.4f} proxy_tau={cfg["tau"]} f1={cfg["f1"]:.3f}')
        del m; cleanup()
    del base, Xp; cleanup()


# ---------------------------------------------------------------- E6
def run_e6(LOG, seed=0):
    arch, attack, rate = ARCH, ATTACK, RATE
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    ipn = ip.cpu().numpy()
    feats = extract_features(base, Xp)
    for det in ['ac', 'ss']:
        if det == 'ac':
            F = activation_clustering(feats, yp)
        else:
            F = spectral_signatures(feats, yp, n_remove=int(ip.sum()))
        dm = detection_metrics(F, ipn)
        for meth in ['retrain_delF', 'scrub', 'recipe']:
            key = dict(exp='E6', arch=arch, attack=attack, rate=rate, seed=seed,
                       recall=dm['recall'], precision=dm['precision'],
                       n_forget=dm['flagged'], method=meth, detector=det)
            if LOG.has(**key):
                continue
            m, t = apply_method(meth, base, Xp, yp, F, attack)
            log_removal(LOG, m, 'E6', arch, attack, rate, meth,
                        dm['recall'], dm['precision'], dm['flagged'], t,
                        notes=f'raw {det} flags; det_rows={dm}')
            del m; cleanup()
    del base, Xp; cleanup()


# ---------------------------------------------------------------- E7b
def run_e7b(LOG, seed=0):
    arch, attack, rate = ARCH, ATTACK, RATE
    base, Xp, yp, ip, _ = poison_and_train(arch, attack, rate, seed)
    n_poison = int(ip.sum())
    mask, pat = _synth_cached(base, Xp, yp, seed, f'{arch}_{attack}')
    true_m = np.zeros((32, 32), bool); true_m[27:31, 27:31] = True
    sm = (mask[0, 0].numpy() > 0.5)
    iou = float((sm & true_m).sum() / max((sm | true_m).sum(), 1))
    key = dict(exp='E7b', arch=arch, attack=attack, rate=rate, seed=seed,
               recall=1.0, precision=1.0, n_forget=n_poison,
               method='nc_synth', detector='none')
    if not LOG.has(**key):
        t0 = time.time()
        m = inoculated_finetune(base, Xp, yp, attack,
                                stamp_fn=lambda z: apply_synth_trigger(z, mask, pat))
        log_removal(LOG, m, 'E7b', arch, attack, rate, 'nc_synth', 1.0, 1.0,
                    n_poison, time.time() - t0, notes=f'synth trigger, IoU={iou:.3f}')
        del m; cleanup()
    del base, Xp; cleanup()


MAIN = ['E1', 'E9', 'E3', 'E6', 'E7b']

if __name__ == '__main__':
    LOG = boot()
    for blk in MAIN:
        print(f'\n===== {blk} =====')
        t0 = time.time()
        {'E1': lambda: run_e1(LOG, 0), 'E9': lambda: run_e9(LOG),
         'E3': lambda: run_e3(LOG, 0), 'E6': lambda: run_e6(LOG, 0),
         'E7b': lambda: run_e7b(LOG, 0)}[blk]()
        print(f'===== {blk} done in {(time.time()-t0)/60:.1f} min =====')
    print(f'ALL DONE -> {LOG.path}')
