"""
FROZEN CORE -- paste as Cell 1 in every team notebook. Do not edit functions.
Your driver cell calls setup(...) then runs your assigned blocks.
Requires: torch, torchvision, sklearn, pandas, numpy. CIFAR-10 attached or Internet ON.
"""
import os, sys, copy, math, time, glob, shutil, random, gc, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# ============================== CONFIG ==============================
TARGET = 0
BS = 512
EPOCHS = 12
LR = {'resnet18': 0.1, 'vgg16_bn': 0.05, 'mobilenet_v2': 0.04}
CLIP = {'mobilenet_v2': 5.0}
OUT = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
PERSON = 'unknown'
SEED = 0
FAST = False
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = dev == 'cuda'
NGPU = torch.cuda.device_count()

def setup(person, seed=0, fast=False):
    """Call once at the top of your driver cell."""
    global PERSON, SEED, FAST, Xtr, ytr, Xte, yte
    PERSON, SEED, FAST = person, seed, fast
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    print(f'[{person}] seed={seed} fast={fast} device={dev} gpus={NGPU} bs={BS} epochs={EPOCHS}')

def set_seed(seed):
    global SEED
    SEED = seed
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup():
    gc.collect()
    if dev == 'cuda':
        torch.cuda.empty_cache()

# ============================== DATA ==============================
Xtr = ytr = Xte = yte = None

def load_cifar(force=False):
    global Xtr, ytr, Xte, yte
    if Xtr is not None and not force:
        return
    os.makedirs('./data', exist_ok=True)
    if not os.path.exists('./data/cifar-10-batches-py'):
        hits = glob.glob('/kaggle/input/**/data_batch_1', recursive=True)
        if hits:
            shutil.copytree(os.path.dirname(hits[0]), './data/cifar-10-batches-py')
            print('CIFAR from', os.path.dirname(hits[0]))
        else:
            tars = glob.glob('/kaggle/input/**/cifar-10-python.tar.gz', recursive=True)
            if tars:
                shutil.copy(tars[0], './data/cifar-10-python.tar.gz')
    tr = torchvision.datasets.CIFAR10('./data', train=True, download=True)
    te = torchvision.datasets.CIFAR10('./data', train=False, download=True)
    if FAST:
        Xtr = torch.tensor(tr.data[:10000]).permute(0, 3, 1, 2).float().to(dev) / 255
        ytr = torch.tensor(tr.targets[:10000]).to(dev)
        Xte = torch.tensor(te.data[:2000]).permute(0, 3, 1, 2).float().to(dev) / 255
        yte = torch.tensor(te.targets[:2000]).to(dev)
    else:
        Xtr = torch.tensor(tr.data).permute(0, 3, 1, 2).float().to(dev) / 255
        ytr = torch.tensor(tr.targets).to(dev)
        Xte = torch.tensor(te.data).permute(0, 3, 1, 2).float().to(dev) / 255
        yte = torch.tensor(te.targets).to(dev)
    del tr, te
    print(f'train {tuple(Xtr.shape)} test {tuple(Xte.shape)}')
    assert FAST or (len(ytr) == 50000 and len(yte) == 10000), 'Full run needs 50k/10k'

# ============================== ATTACKS ==============================
_blend_gen = torch.Generator().manual_seed(123)
BLEND_PATTERN = torch.rand(3, 32, 32, generator=_blend_gen)

def add_trigger(x, attack):
    """Apply the backdoor trigger to a batch of images in [0,1]."""
    x = x.clone().float()
    if attack == 'badnets':
        pat = torch.tensor([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 1]], device=x.device, dtype=x.dtype)
        x[:, :, 27:31, 27:31] = pat
    elif attack == 'blend':
        x = 0.8 * x + 0.2 * BLEND_PATTERN.to(x.device)
    else:
        raise ValueError(f'Unknown attack: {attack}')
    return x

def poison_dataset(X, y, rate, attack, target=TARGET, seed=0):
    """Dirty-label poisoning (deterministic per seed)."""
    g = torch.Generator().manual_seed(seed)
    cand = (y != target).nonzero(as_tuple=False).squeeze(1).cpu()
    n_poison = int(rate * len(y))
    sel = cand[torch.randperm(len(cand), generator=g)[:n_poison]].to(X.device)
    Xp = X.clone()
    yp = y.clone()
    Xp[sel] = add_trigger(X[sel], attack)
    yp[sel] = target
    is_poison = torch.zeros(len(y), dtype=torch.bool, device=X.device)
    is_poison[sel] = True
    return Xp, yp, is_poison

# ============================== MODEL ==============================
class CIFARModel(nn.Module):
    """CIFAR-sized ResNet-18 / VGG-16-BN / MobileNetV2 with features + prune mask."""
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        if arch == 'resnet18':
            m = torchvision.models.resnet18(weights=None)
            m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()
            self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                      m.layer1, m.layer2, m.layer3, m.layer4)
            self.feat_dim = 512
        elif arch == 'vgg16_bn':
            vgg = torchvision.models.vgg16_bn(weights=None)
            self.body = vgg.features
            self.feat_dim = 512
        elif arch == 'mobilenet_v2':
            m = torchvision.models.mobilenet_v2(weights=None)
            m.features[0][0].stride = (1, 1)
            self.body = m.features
            self.feat_dim = 1280
        else:
            raise ValueError(f'Unknown architecture: {arch}')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.feat_dim, 10)
        self.use_amp = (arch != 'mobilenet_v2')
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))
        self.register_buffer('mask', torch.ones(self.feat_dim))

    def extract_features(self, x):
        x = (x - self.mean) / self.std
        h = self.body(x) * self.mask.view(1, -1, 1, 1)
        return self.pool(h).flatten(1)

    def forward(self, x, return_feat=False):
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=(AMP and self.use_amp)):
            p = self.extract_features(x)
            logits = self.fc(p)
        logits, p = logits.float(), p.float()
        return (logits, p) if return_feat else logits

def clone_model(model):
    m = CIFARModel(model.arch)
    m.load_state_dict(copy.deepcopy(model.state_dict()))
    return m.to(dev)

def make_scaler():
    try:
        return torch.amp.GradScaler('cuda', enabled=AMP)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=AMP)

# ============================== AUGMENT ==============================
def augment_batch(x):
    B = x.size(0)
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    oi = torch.randint(0, 9, (B,), device=x.device)
    oj = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (oi[:, None] + ar[None, :])[:, None, :, None]
    cols = (oj[:, None] + ar[None, :])[:, None, None, :]
    bi = torch.arange(B, device=x.device)[:, None, None, None]
    ci = torch.arange(3, device=x.device)[None, :, None, None]
    return xp[bi, ci, rows, cols]

# ============================== TRAIN / EVAL ==============================
def train_model(X, y, arch=None, epochs=None, lr=None, model=None, tag='', aug=True):
    """Train from scratch (or continue a given model). Returns (model, seconds)."""
    epochs = epochs or EPOCHS
    if model is None:
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = CIFARModel(arch).to(dev)
    arch_name = model.arch
    lr = lr or LR[arch_name]
    clip_val = CLIP.get(arch_name)
    net = nn.DataParallel(model) if NGPU > 1 else model
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    total_steps = epochs * math.ceil(len(y) / BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=total_steps, pct_start=0.3)
    scaler = make_scaler()
    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(y), device=dev)
        for k in range(0, len(y), BS):
            idx = perm[k:k + BS]
            if len(idx) < 8:
                continue
            xb = X[idx].float()
            if aug:
                xb = augment_batch(xb)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=(AMP and arch_name != 'mobilenet_v2')):
                loss = F.cross_entropy(net(xb), y[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip_val:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            old = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old:
                sched.step()
        print(f'  [{tag}] ep {ep+1}/{epochs} loss={loss.item():.4f} ({time.time()-t0:.0f}s)')
    if dev == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    model.eval()
    del net, opt, sched, scaler
    cleanup()
    return model, elapsed

@torch.no_grad()
def predict_labels(model, X, bs=512):
    model.eval()
    out = []
    for k in range(0, len(X), bs):
        out.append(model(X[k:k + bs].float()).argmax(1))
    return torch.cat(out)

@torch.no_grad()
def predict_probs(model, X, bs=512):
    model.eval()
    out = []
    for k in range(0, len(X), bs):
        out.append(model(X[k:k + bs].float()).softmax(1))
    return torch.cat(out)

def evaluate(model, attack):
    """Returns (clean_acc %, ASR %) on the test set."""
    preds = predict_labels(model, Xte)
    clean_acc = (preds == yte).float().mean().item() * 100
    nt = (yte != TARGET)
    xt = add_trigger(Xte[nt], attack)
    asr = (predict_labels(model, xt) == TARGET).float().mean().item() * 100
    return round(clean_acc, 2), round(asr, 2)

@torch.no_grad()
def trigger_response(model, attack, n=1000):
    """tau(M): fraction of stamped clean non-target val images predicted target."""
    nt = (yte != TARGET).nonzero(as_tuple=False).squeeze(1)[:n]
    xt = add_trigger(Xte[nt], attack)
    return round((predict_labels(model, xt) == TARGET).float().mean().item() * 100, 2)

@torch.no_grad()
def extract_features(model, X, bs=512, layer='penult'):
    """Penultimate pooled features (CPU tensor), or early-layer features."""
    model.eval()
    body_early = None
    if layer == 'early':
        body_early = nn.Sequential(*list(model.body.children())[:6]).to(dev).eval()
    out = []
    for k in range(0, len(X), bs):
        xb = X[k:k + bs].float()
        if layer == 'penult':
            _, p = model(xb, return_feat=True)
        else:
            h = body_early((xb - model.mean) / model.std)
            p = F.adaptive_avg_pool2d(h, 1).flatten(1)
        out.append(p.cpu())
    return torch.cat(out)

# ============================== DETECTION ==============================
def detection_metrics(flagged, is_poison):
    flagged = np.asarray(flagged, bool); is_poison = np.asarray(is_poison, bool)
    tp = int((flagged & is_poison).sum())
    return dict(flagged=int(flagged.sum()), true_poison=int(is_poison.sum()),
                precision=round(tp / max(int(flagged.sum()), 1), 4),
                recall=round(tp / max(int(is_poison.sum()), 1), 4))

def activation_clustering(feats, y, size_thr=0.35, sil_thr=0.10):
    """AC (Chen et al. 2019): per-class PCA(10) -> 2-means -> flag small cluster."""
    Fn = feats.numpy() if isinstance(feats, torch.Tensor) else np.asarray(feats)
    yn = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    flagged = np.zeros(len(yn), bool)
    for c in range(10):
        ids = np.where(yn == c)[0]
        if len(ids) < 20:
            continue
        z = PCA(n_components=min(10, len(ids), Fn.shape[1]),
                random_state=0).fit_transform(Fn[ids])
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(z)
        counts = np.bincount(km.labels_, minlength=2)
        small = int(np.argmin(counts))
        small_frac = float(counts[small] / counts.sum())
        sil = (silhouette_score(z, km.labels_, sample_size=min(2000, len(ids)),
                                random_state=0)
               if len(set(km.labels_)) > 1 else 0.0)
        if small_frac < size_thr and sil > sil_thr:
            flagged[ids[km.labels_ == small]] = True
    return flagged

def ss_scores(feats, y):
    """Per-sample Spectral Signature score (squared projection on class top SV)."""
    Fn = feats.numpy() if isinstance(feats, torch.Tensor) else np.asarray(feats)
    yn = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    scores = np.zeros(len(yn), np.float64)
    for c in range(10):
        ids = np.where(yn == c)[0]
        if len(ids) < 3:
            continue
        fc = Fn[ids].astype(np.float64)
        fc = fc - fc.mean(axis=0)
        _, _, Vh = np.linalg.svd(fc, full_matrices=False)
        scores[ids] = (fc @ Vh[0]) ** 2
    return scores

def spectral_signatures(feats, y, n_remove):
    scores = ss_scores(feats, y)
    top = np.argsort(scores)[-min(n_remove, len(scores)):]
    flagged = np.zeros(len(scores), bool)
    flagged[top] = True
    return flagged

def auc_score(pos, neg):
    """AUC that pos scores exceed neg scores."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    s = np.concatenate([pos, neg])
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def tpr_at_fpr(pos, neg, fpr=0.10):
    """TPR at fixed FPR (higher score = more poison)."""
    neg = np.sort(np.asarray(neg, float)); pos = np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    thr = np.quantile(neg, 1 - fpr)
    return float((pos >= thr).mean())

# ============================== FORGET SET ==============================
def make_forget_set(is_poison, y, recall, precision, seed):
    """Deterministic simulated detector output F(r,p). Returns (bool mask, stats)."""
    ip = is_poison.cpu().numpy() if isinstance(is_poison, torch.Tensor) else np.asarray(is_poison)
    yn = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    p_idx = np.where(ip)[0]
    n_p = int(round(recall * len(p_idx)))
    n_f = int(round(n_p / max(precision, 1e-9)))
    n_c = max(0, n_f - n_p)
    clean_pool = np.where((~ip) & (yn != TARGET))[0]
    rng = np.random.RandomState(seed * 1009 + 17)
    Fp = rng.choice(p_idx, size=min(n_p, len(p_idx)), replace=False)
    Fc = rng.choice(clean_pool, size=min(n_c, len(clean_pool)), replace=False)
    mask = np.zeros(len(yn), bool); mask[Fp] = True; mask[Fc] = True
    stats = dict(recall=round(len(Fp) / max(len(p_idx), 1), 4),
                 precision=round(len(Fp) / mask.sum(), 4) if mask.sum() else 1.0,
                 n_forget=int(mask.sum()))
    return mask, stats

# ============================== LABEL REPAIR ==============================
def knn_repair(model, X, y, Fset, k=10, layer='penult'):
    """Pseudo-label each flagged sample via kNN over unflagged, non-target data."""
    feats = extract_features(model, X, layer=layer).numpy()
    yn = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    pool = np.where((~Fset) & (yn != TARGET))[0]
    flag = np.where(Fset)[0]
    nn = NearestNeighbors(n_neighbors=min(k, len(pool)), n_jobs=-1).fit(feats[pool])
    dist, idx = nn.kneighbors(feats[flag])
    lab = yn[pool][idx]
    pseudo = np.zeros(len(yn), np.int64)
    for i, f in enumerate(flag):
        w = 1.0 / (dist[i] + 1e-6)
        vote = np.bincount(lab[i], weights=w, minlength=10)
        pseudo[f] = int(vote.argmax())
    return pseudo

def pseudo_accuracy(pseudo, y_true, Fset, is_poison):
    """Diagnostic only: accuracy of pseudo-labels on flagged TRUE poison."""
    ip = is_poison.cpu().numpy() if isinstance(is_poison, torch.Tensor) else np.asarray(is_poison)
    yt = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    m = Fset & ip
    if m.sum() == 0:
        return np.nan
    return float((pseudo[m] == yt[m]).mean())

def refmodel_labels(model, X, y, Fset, epochs=6):
    """Reference-model labeler: train on unflagged data, predict flagged labels."""
    keep = ~Fset
    ref, _ = train_model(X[keep], y[keep], arch=model.arch, epochs=epochs,
                         tag='refmodel', aug=True)
    pred = predict_labels(ref, X[Fset]).cpu().numpy()
    pseudo = np.zeros(len(y), np.int64)
    pseudo[np.where(Fset)[0]] = pred
    return pseudo, ref

# ============================== RECIPE ==============================
def recipe_finetune(model0, X, y, Fset, attack, epochs=3, lr=0.01, bs=None,
                    decouple=False, innoc=True, label_source='penult',
                    ref_model=None, innoc_n=5000, stamp_fn=None):
    """Repair (relabel, keep) + optional inoculation + optional prototype term."""
    bs = bs or BS
    stamp = stamp_fn or (lambda z: add_trigger(z, attack))
    yn = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    if label_source == 'refmodel' and ref_model is not None:
        pseudo_np = predict_labels(ref_model, X[Fset]).cpu().numpy()
        pseudo = np.zeros(len(yn), np.int64); pseudo[np.where(Fset)[0]] = pseudo_np
    else:
        layer = 'early' if label_source == 'early' else 'penult'
        pseudo = knn_repair(model0, X, y, Fset, layer=layer)
    pseudo_t = torch.from_numpy(pseudo).to(dev)

    retain = np.where(~Fset)[0]
    flag = np.where(Fset)[0]
    innoc_idx, X_in = None, None
    if innoc:
        probs = predict_probs(model0, X[retain])
        conf, pred = probs.max(1)
        suspect = ((pred == TARGET) & (conf > 0.9)).cpu().numpy()
        pool = retain[~suspect]
        if len(pool) > innoc_n:
            pool = pool[np.random.RandomState(SEED).choice(len(pool), innoc_n, replace=False)]
        innoc_idx = pool
        X_in = stamp(X[innoc_idx])

    model = clone_model(model0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    n_batches = math.ceil(len(yn) / bs)
    scaler = make_scaler()
    model.train()
    for ep in range(epochs):
        centroids = None
        if decouple:
            sub = retain[np.random.RandomState(SEED + ep).choice(
                len(retain), min(10000, len(retain)), replace=False)]
            with torch.no_grad():
                fz = extract_features(model, X[sub]).to(dev)
                yz = y[sub]
                centroids = torch.stack([
                    fz[yz == c].mean(0) if (yz == c).any()
                    else torch.zeros(fz.shape[1], device=dev) for c in range(10)])
        for b in range(n_batches):
            nf = max(1, int(bs * 0.2))
            nr = int(bs * 0.6) if innoc else bs - nf
            ni = bs - nr - nf
            bi = torch.from_numpy(retain[torch.randint(0, len(retain), (nr,)).numpy()]).to(dev)
            fi = torch.from_numpy(flag[torch.randint(0, len(flag), (nf,)).numpy()]).to(dev)
            xs = [X[bi]]; ys = [y[bi]]
            if innoc:
                sel = torch.randint(0, len(innoc_idx), (ni,)).numpy()
                xs.append(X_in[torch.from_numpy(sel)])
                ys.append(y[torch.from_numpy(innoc_idx[sel])])
            xs.append(X[fi]); ys.append(pseudo_t[fi])
            xb = augment_batch(torch.cat(xs)); yb = torch.cat(ys)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP):
                logits, feat = model(xb, return_feat=True)
                loss = F.cross_entropy(logits, yb)
                if decouple and centroids is not None and len(fi) > 0:
                    d = torch.cdist(feat[-len(fi):], centroids)
                    loss = loss + 0.1 * F.cross_entropy(-d, pseudo_t[fi])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        print(f'  [recipe] ep {ep+1}/{epochs} loss={loss.item():.4f}')
    model.eval()
    cleanup()
    return model

def inoculated_finetune(model0, X, y, attack, epochs=3, lr=0.01, bs=None,
                        innoc_n=5000, stamp_fn=None):
    """Inoculation-only baseline (no label repair)."""
    bs = bs or BS
    stamp = stamp_fn or (lambda z: add_trigger(z, attack))
    retain = np.arange(len(y))
    probs = predict_probs(model0, X)
    conf, pred = probs.max(1)
    suspect = ((pred == TARGET) & (conf > 0.9)).cpu().numpy()
    pool = np.where(~suspect)[0]
    if len(pool) > innoc_n:
        pool = pool[np.random.RandomState(SEED).choice(len(pool), innoc_n, replace=False)]
    X_in = stamp(X[pool])
    model = clone_model(model0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scaler = make_scaler()
    for ep in range(epochs):
        perm = torch.randperm(len(retain), device=dev)
        for k in range(0, len(retain), bs):
            ri = perm[k:k + bs]
            ni = max(1, len(ri) // 4)
            ii = torch.randint(0, len(pool), (ni,), device=dev)
            xb = augment_batch(torch.cat([X[ri], X_in[ii]]))
            yb = torch.cat([y[ri], y[torch.from_numpy(pool).to(dev)[ii]]])
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        print(f'  [inoc] ep {ep+1}/{epochs} loss={loss.item():.4f}')
    model.eval(); cleanup()
    return model

# ============================== BASELINE REMOVERS ==============================
def negrad_plus(model0, X, y, Fset, epochs=3, lr=1e-3, bs=None):
    bs = bs or BS
    retain = np.where(~Fset)[0]; flag = np.where(Fset)[0]
    model = clone_model(model0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for ep in range(epochs):
        for k in range(0, len(flag), bs):
            fi = torch.from_numpy(flag[k:k + bs]).to(dev)
            ri = torch.from_numpy(retain[torch.randint(0, len(retain), (len(fi),)).numpy()]).to(dev)
            loss = -F.cross_entropy(model(X[fi]), y[fi]) + F.cross_entropy(model(X[ri]), y[ri])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); cleanup()
    return model

def scrub_finetune(model0, X, y, Fset, epochs=3, lr=0.01, bs=None, teach_epochs=3):
    """Minimal SCRUB port: teacher = retain-only fine-tune; student = CE(retain) + KL to teacher on Fset."""
    bs = bs or BS
    retain = np.where(~Fset)[0]; flag = np.where(Fset)[0]
    teacher = finetune_retain(model0, X, y, Fset, epochs=teach_epochs, lr=lr)
    teacher.eval()
    model = clone_model(model0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for ep in range(epochs):
        for k in range(0, len(retain), bs):
            ri = torch.from_numpy(retain[k:k + bs]).to(dev)
            fi = torch.from_numpy(flag[torch.randint(0, len(flag), (len(ri),)).numpy()]).to(dev)
            loss_r = F.cross_entropy(model(augment_batch(X[ri])), y[ri])
            with torch.no_grad():
                t = teacher(X[fi]).softmax(1)
            loss_f = F.kl_div(model(X[fi]).log_softmax(1), t, reduction='batchmean')
            loss = loss_r + loss_f
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        print(f'  [scrub] ep {ep+1}/{epochs} loss={loss.item():.4f}')
    model.eval(); cleanup()
    return model

def finetune_retain(model0, X, y, Fset, epochs=3, lr=0.01, bs=None):
    bs = bs or BS
    retain = np.where(~Fset)[0]
    model = clone_model(model0)
    model, _ = train_model(X[retain], y[retain], model=model, epochs=epochs,
                           lr=lr, tag='ft_retain', aug=True)
    return model

def apply_method(method, model0, X, y, Fset, attack, **kw):
    """Uniform dispatcher used by every driver."""
    if method in ('gold', 'retrain_delF'):
        m, t = train_model(X[~Fset], y[~Fset], arch=model0.arch, tag=method, aug=True)
        return m, t
    t0 = time.time()
    if method == 'finetune_retain':
        m = finetune_retain(model0, X, y, Fset)
    elif method == 'negrad_plus':
        m = negrad_plus(model0, X, y, Fset)
    elif method == 'scrub':
        m = scrub_finetune(model0, X, y, Fset)
    elif method == 'nc_style':
        m = inoculated_finetune(model0, X, y, attack)
    elif method == 'recipe':
        m = recipe_finetune(model0, X, y, Fset, attack,
                            decouple=kw.get('decouple', False),
                            innoc=kw.get('innoc', True),
                            label_source=kw.get('label_source', 'penult'),
                            ref_model=kw.get('ref_model'))
    else:
        raise ValueError(method)
    return m, time.time() - t0

# ============================== RELEARNING / MIA ==============================
def relearn_steps(model0, X, y, attack, n=100, max_steps=200, bs=50,
                  lr=0.01, target_asr=90):
    """Fine-tune on n fresh poisoned samples; steps until ASR >= target_asr."""
    g = torch.Generator().manual_seed(999)
    cand = (y != TARGET).nonzero(as_tuple=False).squeeze(1).cpu()
    sel = cand[torch.randperm(len(cand), generator=g)[:n]].to(dev)
    Xf = add_trigger(X[sel], attack); yf = torch.full((n,), TARGET, device=dev)
    model = clone_model(model0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    step = 0
    while step < max_steps:
        idx = torch.randperm(n, device=dev)[:bs]
        loss = F.cross_entropy(model(Xf[idx]), yf[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        step += 1
        if step % 5 == 0:
            _, asr = evaluate(model, attack)
            if asr >= target_asr:
                return step
    return max_steps

@torch.no_grad()
def mia_auc(model, X_train, y_train, X_test, y_test, bs=512, n=5000):
    """Loss-threshold membership inference AUC (train=member)."""
    def losses(X, y):
        out = []
        for k in range(0, min(len(X), n), bs):
            out.append(F.cross_entropy(model(X[k:k + bs]), y[k:k + bs], reduction='none'))
        return torch.cat(out).cpu().numpy()
    return auc_score(losses(X_test, y_test), losses(X_train, y_train))

# ============================== TRIGGER SYNTHESIS ==============================
def synthesize_trigger(model, X, y, steps=200, n=64, lr=0.1, lam=1e-3, seed=0):
    """Neural-Cleanse-style: per-class mask+pattern, keep smallest mask norm."""
    torch.manual_seed(seed)
    model.eval()
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    best = None
    for c in range(10):
        if c == TARGET:
            continue
        ids = (y == c).nonzero(as_tuple=False).squeeze(1)[:n]
        xb = X[ids].clone()
        mask_logit = torch.zeros(1, 1, 32, 32, device=dev, requires_grad=True)
        pat = torch.zeros(1, 3, 32, 32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([mask_logit, pat], lr=lr)
        for s in range(steps):
            m = torch.sigmoid(mask_logit)
            xin = xb * (1 - m) + torch.tanh(pat) * m
            loss = F.cross_entropy(model(xin), torch.full((len(xb),), TARGET, device=dev)) \
                + lam * m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            m = torch.sigmoid(mask_logit).detach()
            norm = float(m.sum())
        if best is None or norm < best['norm']:
            best = dict(norm=norm, src_class=c, mask=m.cpu(), pattern=torch.tanh(pat).detach().cpu())
        if c % 3 == 0:
            print(f'  [synth] class {c} mask_norm={norm:.1f}')
    for p, r in zip(model.parameters(), req):
        p.requires_grad_(r)
    return best

def apply_synth_trigger(x, mask, pattern):
    m = mask.to(x.device); p = pattern.to(x.device)
    return x * (1 - m) + p * m

# ============================== LOGGER ==============================
COLS = ['person', 'exp', 'arch', 'attack', 'rate', 'seed', 'recall', 'precision',
        'n_forget', 'method', 'detector', 'clean_acc', 'asr', 'auc',
        'tpr_at_fpr10', 'prec_at_fpr10', 'n_flags', 'n_true_in_flags',
        'time_sec', 'notes']
KEY = ['exp', 'arch', 'attack', 'rate', 'seed', 'recall', 'precision',
       'n_forget', 'method', 'detector']

# ---------- FROZEN EXPERIMENT GRIDS (all drivers use these) ----------
E1_CONFIGS = [(1.0, 1.0), (0.95, 1.0), (0.90, 1.0), (0.75, 1.0), (0.50, 1.0),
              (0.25, 1.0), (0.90, 0.90), (0.90, 0.70), (0.50, 0.90), (0.50, 0.70)]
E1_METHODS = ['retrain_delF', 'finetune_retain', 'negrad_plus', 'scrub', 'recipe']
E2_RESIDUAL = [2500, 1250, 625, 312, 156, 78, 39, 20, 10, 0]


def poison_and_train(arch, attack, rate, seed, tag='poisoned'):
    """Deterministic poisoned dataset + cached 12-epoch model. Returns (model, Xp, yp, is_poison, sec)."""
    set_seed(seed)
    Xp, yp, ip = poison_dataset(Xtr, ytr, rate, attack, seed=seed)
    ck = f'{OUT}/ckpt_{arch}_{attack}_p{int(rate*100)}_s{seed}.pt'
    if os.path.exists(ck):
        m = CIFARModel(arch).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev))
        m.eval()
        print(f'  loaded {ck}')
        return m, Xp, yp, ip, 0.0
    m, t = train_model(Xp, yp, arch=arch, tag=tag)
    torch.save(m.state_dict(), ck)
    print(f'  saved {ck} ({t:.0f}s)')
    return m, Xp, yp, ip, t

def _r(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ''
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        return round(v, nd)
    return v

class Logger:
    """Append-only CSV with resume: skips rows whose KEY already exists."""
    def __init__(self, path):
        self.path = path
        self.rows = []
        if os.path.exists(path):
            try:
                df = pd.read_csv(path).fillna('')
                self.rows = df.to_dict('records')
                print(f'resumed {path}: {len(self.rows)} rows')
            except Exception as e:
                print('could not resume:', e)
        self.keys = {self._key(r) for r in self.rows}

    @staticmethod
    def _key(r):
        out = []
        for k in KEY:
            v = r.get(k, '')
            if k in ('rate', 'recall', 'precision') and v != '':
                v = round(float(v), 4)
            out.append(v)
        return tuple(out)

    def has(self, **kw):
        return self._key({c: kw.get(c, '') for c in COLS}) in self.keys

    def log(self, **kw):
        row = {c: kw.get(c, '') for c in COLS}
        row['person'] = PERSON
        row['seed'] = kw.get('seed', SEED)
        for c in ['rate', 'recall', 'precision', 'clean_acc', 'asr', 'auc',
                  'tpr_at_fpr10', 'prec_at_fpr10', 'time_sec', 'n_forget',
                  'n_flags', 'n_true_in_flags']:
            row[c] = _r(row[c])
        row['notes'] = str(row['notes']).replace('\n', ' ')[:300]
        k = self._key(row)
        if k in self.keys:
            print('  [log] duplicate skipped:', k)
            return
        self.keys.add(k)
        self.rows.append(row)
        pd.DataFrame(self.rows, columns=COLS).to_csv(self.path, index=False)
        print(f'  [log] {row["exp"]}|{row["method"]}|r={row["recall"]}|p={row["precision"]} '
              f'| acc={row["clean_acc"]} asr={row["asr"]}')

# ============================== CONVENIENCE WRAPPERS ==============================
def log_removal(logger, model, exp, arch, attack, rate, method, recall, precision,
                n_forget, t, notes=''):
    acc, asr = evaluate(model, attack)
    logger.log(exp=exp, arch=arch, attack=attack, rate=rate, recall=recall,
               precision=precision, n_forget=n_forget, method=method,
               detector='none', clean_acc=acc, asr=asr, time_sec=t, notes=notes)
    return acc, asr

def log_detection(logger, exp, arch, attack, rate, model, X, y, is_poison,
                  detector='ss', n_remove=None, recall='', precision='',
                  n_forget='', notes=''):
    """Logs a detector evaluation on the CURRENT model. AUC = poison vs clean class-0."""
    feats = extract_features(model, X)
    ynp = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    ip = is_poison.cpu().numpy() if isinstance(is_poison, torch.Tensor) else np.asarray(is_poison)
    scores = ss_scores(feats, y)
    c0 = ynp == TARGET
    pos = ip & c0; neg = (~ip) & c0
    if detector == 'ac':
        flag = activation_clustering(feats, y)
        auc = np.nan; tpr = np.nan; prec = np.nan
    else:
        nr = n_remove or int(1.5 * ip.sum())
        top = np.argsort(scores)[-min(nr, len(scores)):]
        flag = np.zeros(len(scores), bool); flag[top] = True
        auc = auc_score(scores[pos], scores[neg])
        tpr = tpr_at_fpr(scores[pos], scores[neg])
        thr = np.quantile(scores[neg], 0.90) if neg.any() else np.inf
        det = c0 & (scores >= thr)
        prec = float((det & ip).sum() / max(det.sum(), 1))
    m = detection_metrics(flag, ip)
    logger.log(exp=exp, arch=arch, attack=attack, rate=rate, recall=recall,
               precision=precision, n_forget=n_forget, method='poisoned',
               detector=detector, auc=auc, tpr_at_fpr10=tpr, prec_at_fpr10=prec,
               n_flags=m['flagged'], n_true_in_flags=int((flag & ip).sum()),
               notes=notes)
    return flag, scores, m
