"""Recompute every quantitative claim used in the paper from the frozen CSVs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import build_paper_assets as B

D = B.load()
e1 = D["e1"]
e1["precision"] = e1["precision"].round(2)
cells = [(1.0, 1.0), (0.95, 1.0), (0.9, 1.0), (0.75, 1.0), (0.5, 1.0), (0.25, 1.0),
         (0.9, 0.9), (0.9, 0.7), (0.5, 0.9), (0.5, 0.7)]
methods = ["retrain_delF", "finetune_retain", "negrad_plus", "scrub", "recipe"]
gold = {}
for s in (0, 1, 2):
    d = e1[(e1.method == "retrain_delF") & (e1.seed == s) & np.isclose(e1.recall, 1.0)]
    gold[s] = float(d.clean_acc.iloc[0])
print("GOLD", gold, "mean", np.mean(list(gold.values())))

print("\nE1 CELLS (n, forget, ASR mean/sd, clean mean/sd, per-seed asr)")
for m in methods:
    print("--", m)
    for r, p in cells:
        d = e1[(e1.method == m) & np.isclose(e1.recall, r) & np.isclose(e1.precision, p)].sort_values("seed")
        if len(d) == 0:
            print(f"  ({r},{p}): MISSING")
            continue
        print(f"  ({r:.2f},{p:.1f}) n={len(d)} |F|={int(d.n_forget.iloc[0])} "
              f"ASR {d.asr.mean():.2f}+-{d.asr.std(ddof=1):.2f} {list(np.round(d.asr.values,2))} "
              f"clean {d.clean_acc.mean():.2f}+-{d.clean_acc.std(ddof=1):.2f} {list(np.round(d.clean_acc.values,2))}")

print("\nNC STYLE", e1[e1.method == "nc_style"].asr.mean(), e1[e1.method == "nc_style"].clean_acc.mean(),
      "n", len(e1[e1.method == "nc_style"]), list(np.round(e1[e1.method == 'nc_style'].asr.values, 2)))

print("\nSUCCESS COUNTS (mean rule r*)")
for m in methods:
    succ = []
    for r, p in cells:
        n = 0
        for s in (0, 1, 2):
            d = e1[(e1.method == m) & (e1.seed == s) & np.isclose(e1.recall, r) & np.isclose(e1.precision, p)]
            if len(d) and d.asr.iloc[0] <= 5.0 and d.clean_acc.iloc[0] >= gold[s] - 2.0:
                n += 1
        succ.append(n)
    print(m, succ)
    # mean-rule r*
    rstar = None
    for r, p in cells:
        if p != 1.0:
            continue
        d = e1[(e1.method == m) & np.isclose(e1.recall, r) & np.isclose(e1.precision, p)]
        if len(d) and d.asr.mean() <= 5.0 and d.clean_acc.mean() >= np.mean(list(gold.values())) - 2.0:
            rstar = r if rstar is None else min(rstar, r)
    print("  mean-rule r*:", rstar)

print("\nE2 (urja)")
print(D["urja"][["arch", "attack", "f", "ASR", "clean_acc"]].to_string(index=False))
print("E2 ASR range", D["urja"].ASR.min(), D["urja"].ASR.max(),
      "ASR at f=1", D["urja"][np.isclose(D["urja"].f, 1.0)].ASR.min(), D["urja"][np.isclose(D["urja"].f, 1.0)].ASR.max(),
      "clean range", D["urja"].clean_acc.min(), D["urja"].clean_acc.max())
print("clean within setting delta max:")
for (a, atk), g in D["urja"].groupby(["arch", "attack"]):
    print(" ", a, atk, round(g.clean_acc.max() - g.clean_acc.min(), 2))

print("\nE3 CaR")
e3 = D["e3"]
cp = e3[e3.method == "car_proxy"].copy()
taus = cp.notes.str.extract(r"tau=([\d.]+)")[0].astype(float)
print("n thresholds", len(cp), "tau min/max", taus.min(), taus.max(), "n==100", (taus == 100).sum())
print(cp[["recall", "precision", "clean_acc", "asr"]].describe().to_string())
f1 = e3[e3.method == "f1_full"].iloc[0]
print("F1 row", f1.recall, f1.precision, f1.clean_acc, f1.asr, f1.notes)

print("\nE6")
print(D["e6"][["method", "recall", "precision", "clean_acc", "asr"]].to_string(index=False))

print("\nE7")
print(D["e7"][["arch", "attack", "method", "recall", "clean_acc", "asr"]].to_string(index=False))
for (a, atk), g in D["e7"].groupby(["arch", "attack"]):
    gg = g[g.method == "gold"].iloc[0]
    for m in ["retrain_delF", "recipe"]:
        for r in [0.75, 0.5]:
            x = g[(g.method == m) & np.isclose(g.recall, r)]
            if len(x):
                print(f"  {a}/{atk} {m} r={r}: ASR {x.asr.iloc[0]:.2f} clean {x.clean_acc.iloc[0]:.2f} "
                      f"delta {x.clean_acc.iloc[0]-gg.clean_acc:.2f}")

print("\nE7b")
print(D["e7b"].to_string(index=False))

print("\nE8")
print(D["e8"][["variant", "method", "clean_acc", "asr"]].to_string(index=False))

print("\nE9")
print(D["e9"][["seed", "method", "clean_acc", "asr", "auc"]].to_string(index=False))

print("\nSPIKE")
sp = D["spike"]
print(sp[["tag", "asr", "clean", "auc_missed_vs_clean0"]].to_string(index=False))

print("\nSISA")
print(D["sisa"][["recall", "clean_acc", "asr"]].to_string(index=False))

print("\nFINE PRUNING")
print(D["fp"][["arch", "prune_ratio", "clean_acc", "ASR"]].to_string(index=False))

print("\nPILOT rows")
print(D["pilot"][["method", "clean_acc", "ASR"]].to_string(index=False))

print("\nDURABILITY")
print(D["rel"].to_string(index=False))
print("\nURJA metrics")
print(D["urja"][["arch", "attack", "method", "mia_auc", "kl_clean", "kl_trig"]].to_string(index=False))

print("\nDETECTOR")
print(D["det"].to_string(index=False))

print("\nRF")
rf = D["rf"]
rfs = D["rfs"]
print("base:", rf[rf.method == "base"][["seed", "clean_acc", "asr"]].to_string(index=False))
print("base means", rf[rf.method == "base"].asr.mean(), rf[rf.method == "base"].clean_acc.mean())
print("\nRF table cells")
for m in ["gold_retrain", "finetune_retain", "recipe", "nc_style"]:
    for r in [1.0, 0.75, 0.5]:
        d = rf[(rf.method == m) & np.isclose(rf.recall, r) & np.isclose(rf.precision, 1.0)]
        if len(d):
            print(f"  {m} r={r}: n={len(d)} ASR {d.asr.mean():.2f}+-{d.asr.std(ddof=1):.2f} "
                  f"clean {d.clean_acc.mean():.2f}+-{d.clean_acc.std(ddof=1):.2f}")
    print(f"  {m} clean all: {rf[rf.method==m].clean_acc.mean():.2f}+-{rf[rf.method==m].clean_acc.std(ddof=1):.2f}")

print("\nRF precision .7 cell (r=.75)")
for m in ["gold_retrain", "finetune_retain", "recipe", "nc_style"]:
    d = rf[(rf.method == m) & np.isclose(rf.recall, 0.75) & np.isclose(rf.precision, 0.7)]
    print(" ", m, len(d), "ASR", round(d.asr.min(), 2), "-", round(d.asr.max(), 2),
          "mean", round(d.asr.mean(), 2))

print("\nRF per-SNR base clean at key SNRs")
b = rfs[rfs.method == "base"].groupby("snr")[["clean_acc", "asr"]].mean()
print(b.to_string())
print("base ASR min/max", b.asr.min(), b.asr.max())

print("\nRF fine-tune per-SNR")
for r in [1.0, 0.75, 0.5]:
    d = rfs[(rfs.method == "finetune_retain") & np.isclose(rfs.recall, r) & np.isclose(rfs.precision, 1.0)]
    g = d.groupby("snr").asr.mean()
    print(f" r={r}: min {g.min():.2f} at {g.idxmin()} max {g.max():.2f}")

print("\nRF recipe r=.5 per-SNR")
for m in ["recipe", "nc_style"]:
    d = rfs[(rfs.method == m) & np.isclose(rfs.recall, 0.5) & np.isclose(rfs.precision, 1.0)]
    g = d.groupby("snr").asr.mean()
    print(m, dict(g.round(2)))

print("\nRF deletion partial per-SNR")
for r in [0.75, 0.5]:
    d = rfs[(rfs.method == "gold_retrain") & np.isclose(rfs.recall, r) & np.isclose(rfs.precision, 1.0)]
    g = d.groupby("snr").asr.mean()
    print(f" r={r}: min {g.min():.2f} max {g.max():.2f}")

print("\nGAMMA star")
print(D["gam"].to_string(index=False))

print("\nRF per-SNR clean, base/gold/recipe r=1")
for m in ["base", "gold_retrain", "recipe"]:
    d = rfs[(rfs.method == m) & (np.isclose(rfs.recall, 1.0) | (rfs.method == "base"))]
    g = d.groupby("snr").clean_acc.mean()
    print(m, dict(g.round(2)))

print("\nRF clean at -20,-16,-10,-6,-4,0,4,18")
for m in ["base", "gold_retrain", "recipe"]:
    d = rfs[(rfs.method == m)]
    g = d.groupby("snr").clean_acc.mean()
    print(m, {s: round(g[s], 2) for s in [-20, -16, -10, -6, -4, 0, 4, 18]})
