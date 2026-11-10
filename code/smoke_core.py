"""CPU functional smoke test for core_kaggle.py (no CIFAR download)."""
import os, sys
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_kaggle as C

torch.set_num_threads(2)
C.dev = 'cpu'
C.AMP = False
C.BS = 32
C.setup('smoke', seed=0)

# --- attacks / poisoning ---
x = torch.rand(8, 3, 32, 32)
xt = C.add_trigger(x, 'badnets')
assert xt.shape == x.shape and float(xt[:, :, 27, 27].max()) == 1.0
xb = C.add_trigger(x, 'blend')
assert torch.allclose(xb, 0.8 * x + 0.2 * C.BLEND_PATTERN)

y = torch.randint(0, 10, (400,))
X = torch.rand(400, 3, 32, 32)
Xp, yp, ip = C.poison_dataset(X, y, 0.05, 'badnets', seed=0)
assert int(ip.sum()) == 20 and int((yp[ip] == 0).all())
Xp2, yp2, ip2 = C.poison_dataset(X, y, 0.05, 'badnets', seed=0)
assert torch.equal(ip, ip2)

# --- forget set math ---
F, st = C.make_forget_set(ip, yp, recall=0.5, precision=0.7, seed=0)
assert st['n_forget'] == 14, st          # 10 poison + 4 clean (20 true poison total)
assert abs(st['recall'] - 0.5) < 1e-6 and abs(st['precision'] - 10/14) < 1e-3, st
assert not (F & ~ip.cpu().numpy() & (yp.cpu().numpy() == 0)).any(), 'clean class-0 leaked into F'

# --- detection metrics ---
pos, neg = np.array([3., 4., 5.]), np.array([1., 2., 2.5])
assert C.auc_score(pos, neg) == 1.0
assert C.tpr_at_fpr(pos, neg, 0.1) == 1.0
assert C.auc_score([], neg) != C.auc_score([], neg)  # nan

# --- model / features / augment ---
m = C.CIFARModel('resnet18').to('cpu')
lg, feat = m(torch.rand(4, 3, 32, 32), return_feat=True)
assert lg.shape == (4, 10) and feat.shape == (4, 512)
ab = C.augment_batch(torch.rand(6, 3, 32, 32))
assert ab.shape == (6, 3, 32, 32)
fe = C.extract_features(m, torch.rand(8, 3, 32, 32), bs=4, layer='early')
assert fe.shape[0] == 8 and fe.shape[1] == 128, fe.shape

# --- kNN repair + pseudo accuracy ---
pseudo = C.knn_repair(m, X, y, F)
assert pseudo.shape == (400,) and pseudo[F].min() >= 0
acc_diag = C.pseudo_accuracy(pseudo, y, F, ip)
assert 0 <= acc_diag <= 1

# --- removers on tiny data ---
C.SEED = 0
t0 = torch.rand(64, 3, 32, 32)
y0 = torch.randint(0, 10, (64,))
ip0 = torch.zeros(64, dtype=torch.bool); ip0[:3] = True; y0[:3] = 0
F0, _ = C.make_forget_set(ip0, y0, 0.66, 1.0, 0)
for fn in [C.finetune_retain, C.negrad_plus, C.scrub_finetune,
           lambda mo, X_, y_, F_: C.inoculated_finetune(mo, X_, y_, 'badnets', innoc_n=20)]:
    out = fn(m, t0, y0, F0)
    assert isinstance(out, C.CIFARModel)
out = C.recipe_finetune(m, t0, y0, F0, 'badnets', epochs=1, bs=16, innoc=True,
                        decouple=True, innoc_n=20)
assert isinstance(out, C.CIFARModel)
out2, dt = C.apply_method('recipe', m, t0, y0, F0, 'badnets')
assert isinstance(out2, C.CIFARModel)

# --- trigger synthesis (2 steps) ---
best = C.synthesize_trigger(m, t0, y0, steps=2, n=4)
assert best is not None and 'mask' in best
st_img = C.apply_synth_trigger(torch.rand(2, 3, 32, 32), best['mask'], best['pattern'])
assert st_img.shape == (2, 3, 32, 32)

# --- relearn / mia ---
steps = C.relearn_steps(m, t0, y0, 'badnets', n=8, max_steps=4, bs=4)
assert 0 < steps <= 4
ma = C.mia_auc(m, t0[:32], y0[:32], t0[32:], y0[32:], n=32)
assert ma == ma

# --- logger roundtrip ---
p = 'smoke_runs.csv'
if os.path.exists(p):
    os.remove(p)
L = C.Logger(p)
L.log(exp='E1', arch='resnet18', attack='badnets', rate=0.05, seed=0, recall=0.5,
      precision=1.0, n_forget=10, method='recipe', detector='none',
      clean_acc=90.0, asr=1.0, time_sec=3.2, notes='smoke')
L2 = C.Logger(p)
assert len(L2.rows) == 1
L2.log(exp='E1', arch='resnet18', attack='badnets', rate=0.05, seed=0, recall=0.5,
       precision=1.0, n_forget=10, method='recipe', detector='none',
       clean_acc=90.0, asr=1.0, time_sec=3.2, notes='dup')
assert len(L2.rows) == 1, 'resume dedupe failed'
os.remove(p)
print('SMOKE PASS')
