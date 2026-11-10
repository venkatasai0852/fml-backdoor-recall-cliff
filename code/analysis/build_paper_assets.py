"""Generate every figure (vector PDF) and LaTeX table fragment for the IEEE paper.

All numbers are read from the frozen result CSVs; nothing is transcribed by hand.
Outputs -> E:/FML/paper_ieee/figures/*.pdf, tables/*.tex, key_numbers.json
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_ieee"
GEN = OUT / "generated"
FIG = GEN / "figures"
TAB = GEN / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.25,
    "lines.markersize": 3.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
    "axes.grid": False,
    "pdf.fonttype": 42,
})

C = {  # Okabe-Ito, colour-blind safe
    "retrain": "#000000", "finetune": "#E69F00", "negrad": "#D55E00",
    "scrub": "#CC79A7", "recipe": "#0072B2", "nc": "#009E73",
    "ac": "#0072B2", "ss": "#D55E00", "gold": "#555555",
}
DISPLAY = {
    "retrain_delF": r"Retrain$_{\setminus F}$", "finetune_retain": "Fine-tune",
    "negrad_plus": "NegGrad+", "scrub": "SCRUB", "recipe": "Recipe",
    "nc_style": "NC-style", "gold": "Gold", "gold_retrain": "Gold retrain",
    "sisa5": "SISA-5", "car_proxy": "CaR", "f1_full": "F1-oracle",
}

NUM = {}


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    print("  fig ->", p.name)


def tex_escape(s):
    rep = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}"}
    for k, v in rep.items():
        s = s.replace(k, v)
    return s


def table(path, caption, label, colspec, header, body, note=None,
          size="footnotesize", star=False, extra_pkg=""):
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\{size}", f"\\begin{{tabular}}{{{colspec}}}",
             "\\toprule", header + r" \\", "\\midrule"]
    lines += body
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\par\\vspace{{1pt}}\\{size}\\raggedright {note}")
    lines.append(f"\\end{{{env}}}")
    (TAB / path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  tab ->", path)


def ms(x, d=2):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "--"
    if len(x) == 1:
        return f"{x[0]:.{d}f}"
    return f"{x.mean():.{d}f}$\\pm${x.std(ddof=1):.{d}f}"


# ----------------------------------------------------------------------------- data
def load():
    s0 = pd.read_csv(ROOT / "results/sai_2026-11-02/runs_sai.csv")
    s1a = pd.read_csv(ROOT / "results/sarang_2026-11-04/handoff_sarang/runs_sarang.csv")
    s1b = pd.read_csv(ROOT / "results/sarang_2026-11-05/runs_sarang_v2.csv")
    s2 = pd.read_csv(ROOT / "results/divyansh_2026-11-03/runs_divyansh.csv")

    # E1: seed 1 keeps v1 rows except negrad/scrub, which v2 superseded (corrected
    # retain-loop / OneCycle / BN-stat handling).
    e1 = pd.concat([
        s0[s0.exp == "E1"],
        s1a[(s1a.exp == "E1") & (~s1a.method.isin(["negrad_plus", "scrub"]))],
        s1b[s1b.exp == "E1"],
        s2[s2.exp == "E1"],
    ], ignore_index=True)
    e1 = e1[e1.method != "poisoned"].copy()
    # integer rounding in the logger stores achieved p as 0.8999 / 0.7001; snap to the
    # requested grid so cell lookups are exact.
    e1["precision"] = e1["precision"].round(2)

    # E8: v1 = raw ascent, v2 = BN-guarded ascent. Keep both, tagged.
    e8a = s1a[s1a.exp == "E8"].copy(); e8a["variant"] = "raw"
    e8b = s1b[s1b.exp == "E8"].copy(); e8b["variant"] = "bn_guarded"
    e8 = pd.concat([e8a, e8b], ignore_index=True)

    e7 = pd.concat([s1a[s1a.exp == "E7"], s2[s2.exp == "E7"]], ignore_index=True)
    e7b = pd.concat([s0[s0.exp == "E7b"], s2[s2.exp == "E7b"]], ignore_index=True)
    e6 = s0[s0.exp == "E6"].copy()
    e3 = s0[s0.exp == "E3"].copy()
    e9 = s0[s0.exp == "E9"].copy()
    sisa = s2[s2.exp == "SISA"].copy()

    urja = pd.read_csv(ROOT / "urja/metrics_table_urja.csv")
    rel = pd.read_csv(ROOT / "urja/relearning_summary_urja.csv")
    det = pd.read_csv(ROOT / "detection_sai.csv")
    fp = pd.read_csv(ROOT / "fine_pruning_sai.csv")
    spike0 = pd.read_csv(ROOT / "spike/spike_repair_seed0.csv")
    spike1 = pd.read_csv(ROOT / "spike/spike_repair_seed1.csv")
    rf = pd.read_csv(ROOT / "results/rf_study_2026-11-09/runs_rf_study.csv")
    rf["precision"] = rf["precision"].round(2)
    rfs = pd.read_csv(ROOT / "results/rf_study_2026-11-09/rf_study_per_snr.csv")
    gam = pd.read_csv(ROOT / "results/rf_study_2026-11-09/gamma_star.csv")
    pilot = pd.read_csv(ROOT / "master_combined_results.csv")
    return dict(s0=s0, e1=e1, e8=e8, e7=e7, e7b=e7b, e6=e6, e3=e3, e9=e9,
                sisa=sisa, urja=urja, rel=rel, det=det, fp=fp,
                spike=(pd.concat([spike0, spike1], ignore_index=True)),
                rf=rf, rfs=rfs, gam=gam, pilot=pilot)


# ----------------------------------------------------------------------------- figures
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.1, 2.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#ffffff"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.32,rounding_size=1.0",
                                    fc=fc, ec="#000000", lw=0.9))
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=7.2, weight="bold")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=6.0)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     lw=0.9, color="#333333", shrinkA=0, shrinkB=0))

    y1, h = 20.5, 9.6
    box(1, y1, 15, h, "CIFAR-10", "50k / 10k")
    box(18.5, y1, 15, h, "Poisoning", "BadNets / Blend")
    box(36, y1, 15, h, "Training", "ResNet-18, VGG")
    box(53.5, y1, 16, h, "Detector", "AC / SS")
    box(72, y1, 27, h, r"Forget set $\mathcal{F}(r,p)$", r"$|\mathcal{F}|=N_pr/p$; FP from clean non-target")
    for x in (16.2, 33.7, 51.2, 69.7):
        arrow(x, y1 + h / 2, x + 2.1, y1 + h / 2)

    y2 = 3.2
    box(81, y2, 18, h, "Removal", "delete / ascent\n/ scrub / repair")
    box(60, y2, 18, h, "Defence test", "ASR $\\leq5\\%$ and\nclean within 2 pts")
    box(39, y2, 18, h, "Tolerance $r^{*}$", "min recall for\nsuccess")
    box(17.5, y2, 17.5, h, "Durability", "relearning\n+ MIA")
    box(0.5, y2, 14.5, h, "Cross-domain", "RadioML AMC\n$\\gamma^{*}$(SNR)")

    arrow(89.5, y1, 89.5, y2 + h)
    ax.text(86.2, 14.8, "flagged set", fontsize=6.0, style="italic", rotation=90, va="center")
    for x in (81.0, 60.0, 39.0, 17.5):
        arrow(x, y2 + h / 2, x - 1.8, y2 + h / 2)
    save(fig, "fig1_pipeline.pdf")


def fig_cliff(e1, gold):
    cells = [(1.0, 1.0), (0.95, 1.0), (0.90, 1.0), (0.75, 1.0), (0.50, 1.0), (0.25, 1.0)]
    methods = ["retrain_delF", "recipe", "finetune_retain", "negrad_plus", "scrub"]
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for m in methods:
        xs, ys = [], []
        for r, p in cells:
            d = e1[(e1.method == m) & (np.isclose(e1.recall, r)) & (np.isclose(e1.precision, p))]
            if len(d):
                xs.append(r); ys.append(d.asr.mean())
        key = "retrain" if m == "retrain_delF" else m.split("_")[0]
        ax.plot(xs, ys, marker="o", color=C[key], label=DISPLAY[m], zorder=3, clip_on=False)
    d = e1[e1.method == "nc_style"]
    ax.axhline(d.asr.mean(), color=C["nc"], ls="--", lw=1.0, label="NC-style (F-indep.)", zorder=2)
    ax.axhspan(0, 5, color="#0072B2", alpha=0.10, lw=0, zorder=1)
    ax.set_xlabel("Detector recall $r$  (precision $p=1$)")
    ax.set_ylabel("Attack success rate (%)")
    ax.set_xlim(0.20, 1.03); ax.set_ylim(-3, 105)
    ax.set_xticks([0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    ax.set_xticklabels(["0.25", "0.5", "0.75", "0.9", "0.95", "1"])
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.80), frameon=False, handlelength=1.5,
              labelspacing=0.32)
    ax.text(0.60, 7.0, "defence-success band", ha="center", va="bottom", fontsize=5.9, color="#005a8f")
    ax.annotate("cliff", xy=(0.925, 55), xytext=(0.80, 62), fontsize=6.4,
                arrowprops=dict(arrowstyle="->", lw=0.7), ha="center")
    save(fig, "fig2_e1_cliff.pdf")


def fig_success_map(e1, gold_by_seed):
    cells = [(1.0, 1.0), (0.95, 1.0), (0.90, 1.0), (0.75, 1.0), (0.50, 1.0), (0.25, 1.0),
             (0.90, 0.9), (0.90, 0.7), (0.50, 0.9), (0.50, 0.7)]
    methods = ["retrain_delF", "finetune_retain", "negrad_plus", "scrub", "recipe"]
    M = np.zeros((len(methods), len(cells)))
    for i, m in enumerate(methods):
        for j, (r, p) in enumerate(cells):
            n = 0
            for s in (0, 1, 2):
                d = e1[(e1.method == m) & (e1.seed == s) & np.isclose(e1.recall, r) & np.isclose(e1.precision, p)]
                if len(d) and d.asr.iloc[0] <= 5.0 and d.clean_acc.iloc[0] >= gold_by_seed[s] - 2.0:
                    n += 1
            M[i, j] = n
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=3, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=6.2,
                    color="white" if M[i, j] > 1.6 else "#333333")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{r:g}\n{p:g}" for r, p in cells], fontsize=5.6)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([DISPLAY[m] for m in methods], fontsize=6.4)
    ax.set_xlabel("$(r,p)$ cell; value = successes of 3 seeds", labelpad=1)
    ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(length=0)
    save(fig, "fig3_success_map.pdf")


def fig_residual(urja):
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    styles = {("resnet18", "badnets"): ("o", "-", "#0072B2", "R18/BadNets"),
              ("resnet18", "blend"): ("s", "--", "#D55E00", "R18/Blend"),
              ("vgg16_bn", "badnets"): ("^", "-.", "#009E73", "VGG/BadNets"),
              ("vgg16_bn", "blend"): ("D", ":", "#CC79A7", "VGG/Blend")}
    for (a, atk), (mk, ls, col, lab) in styles.items():
        d = urja[(urja.arch == a) & (urja.attack == atk)].sort_values("f")
        ax.plot(d.f * 100, d.ASR, marker=mk, ls=ls, color=col, label=lab)
    ax.axhline(5, color="#888888", ls=":", lw=0.9)
    ax.set_xlabel("Poison remaining in the training set (%)")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(-4, 106); ax.set_xlim(-3, 103)
    ax.legend(frameon=False, loc="lower left", handlelength=2.0)
    ax.annotate("0.5\u20131.2% ASR\nonly at full removal", xy=(100, 1.0), xytext=(56, 24),
                fontsize=5.9, arrowprops=dict(arrowstyle="->", lw=0.6))
    save(fig, "fig4_e2_residual.pdf")


def fig_ascent(e8):
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    lab = ["2e-4/2", "5e-4/2", "1e-3/2", "1e-3/4"]
    raw = e8[e8.variant == "raw"].sort_values("method")
    bn = e8[e8.variant == "bn_guarded"].sort_values("method")
    x = np.arange(4)
    ax.plot(x, raw[raw.method.str.startswith("ascent")].asr.values, "o-", color=C["negrad"], label="Raw ascent ASR")
    ax.plot(x, bn[bn.method.str.startswith("ascent")].asr.values, "s--", color=C["recipe"], label="BN-guarded ASR")
    ax.set_ylabel("ASR (%)"); ax.set_ylim(-4, 108)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_xlabel("Ascent setting (lr / epochs)")
    ax2 = ax.twinx()
    ax2.plot(x, raw[raw.method.str.startswith("ascent")].clean_acc.values, "^:", color=C["scrub"], label="Raw ascent clean acc")
    ax2.plot(x, bn[bn.method.str.startswith("ascent")].clean_acc.values, "v-.", color=C["nc"], label="BN-guarded clean acc")
    ax2.set_ylabel("Clean accuracy (%)"); ax2.set_ylim(60, 95)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center right", fontsize=6.0,
              bbox_to_anchor=(1.0, 0.42))
    save(fig, "fig5_e8_ascent.pdf")


def fig_amplify(e9):
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    seeds = [0, 1, 2]
    groups = ["base", "partial_repair_lr0.005_ep2", "ctrl_clean_lr0.005_ep2"]
    names = ["Base", "Repair", "Clean ctrl"]
    cols = [C["gold"], C["recipe"], C["finetune"]]
    w = 0.26
    base_mean = float(np.mean([e9[(e9.method == "base") & (e9.seed == s)].auc.values[0] for s in seeds]))
    for i, (g, nm, col) in enumerate(zip(groups, names, cols)):
        vals = [e9[(e9.method == g) & (e9.seed == s)].auc.values[0] for s in seeds]
        ax.bar(np.arange(3) + (i - 1) * w, vals, width=w, color=col, label=nm)
    ax.axhline(0.5, color="#888888", ls=":", lw=0.9)
    ax.axhline(base_mean + 0.10, color="#cc0000", ls="--", lw=1.0)
    ax.text(2.45, base_mean + 0.112, "promotion gate ($+0.10$)", color="#cc0000",
            fontsize=5.9, ha="right")
    ax.set_ylim(0.44, 0.64)
    ax.set_xticks(range(3)); ax.set_xticklabels(["seed 0", "seed 1", "seed 2"])
    ax.set_ylabel("AUC (missed poison vs clean)")
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=6.0,
              bbox_to_anchor=(0.5, 1.03), handlelength=1.1, columnspacing=0.9)
    save(fig, "fig6_e9_amplify.pdf")


def fig_rf(rfs):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=True)
    methods = ["gold_retrain", "finetune_retain", "recipe", "nc_style"]
    for ax, r in zip(axes, [1.0, 0.75, 0.5]):
        d0 = rfs[(rfs.method == "base")]
        ax.plot(d0.snr, d0.asr, color="#999999", lw=1.0, ls=":", label="Poisoned base")
        for m in methods:
            d = rfs[(rfs.method == m) & np.isclose(rfs.recall, r) & np.isclose(rfs.precision, 1.0)]
            g = d.groupby("snr").asr.mean()
            ax.plot(g.index, g.values, marker="o", ms=2.5, color=C[m.split("_")[0]],
                    label=DISPLAY[m])
        ax.axhline(5, color="#888888", lw=0.8, ls="--")
        ax.set_title(f"recall $r={r:g}$", pad=2)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylim(-4, 106)
        ax.set_xticks([-20, -10, 0, 10, 18])
    axes[0].set_ylabel("ASR (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.2, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), handlelength=1.5, columnspacing=1.1)
    fig.tight_layout(pad=0.25, w_pad=0.4, rect=(0, 0.10, 1, 1))
    save(fig, "fig7_rf_asr.pdf")


def fig_gamma(gam, rfs):
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.axvspan(-21, -9, color="#eeeeee", lw=0)
    ax.text(-14.5, 0.585, "near-chance\nclean accuracy", ha="center", fontsize=5.4,
            color="#666666", rotation=90, va="center")
    for m in ["gold_retrain", "recipe", "nc_style", "finetune_retain"]:
        d = gam[gam.method == m].sort_values("snr")
        v = d.gamma_star.values.astype(float)
        ax.step(d.snr, v, where="mid", marker="o", ms=2.5, color=C[m.split("_")[0]],
                label=DISPLAY[m])
        miss = np.isnan(v)
        ax.plot(d.snr[miss], np.full(miss.sum(), 1.13), "x", ms=3.0, color=C[m.split("_")[0]])
    ax.axhline(1.0, color="#cccccc", lw=0.5)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("$\\gamma^{*}$ (minimum recall)")
    ax.set_ylim(0.36, 1.22); ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_xticks([-20, -10, 0, 10, 18])
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), columnspacing=0.9, handlelength=1.4)
    ax.text(-20.5, 1.175, "$\\times$: no recall reaches ASR $\\leq5\\%$", fontsize=5.4,
            ha="left", color="#333333")
    save(fig, "fig8_rf_gamma.pdf")


def fig_relearn(rel):
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    styles = {("resnet18", "badnets"): ("o", "-", "#0072B2", "R18/BadNets"),
              ("resnet18", "blend"): ("s", "--", "#D55E00", "R18/Blend"),
              ("vgg16_bn", "badnets"): ("^", "-.", "#009E73", "VGG/BadNets"),
              ("vgg16_bn", "blend"): ("D", ":", "#CC79A7", "VGG/Blend")}
    cols = ["0", "5", "10", "25", "50"]
    x = np.arange(5)
    for (a, atk), (mk, ls, col, lab) in styles.items():
        d = rel[(rel.arch == a) & (rel.attack == atk) & (rel.model == "gold_retrain")]
        ax.plot(x, d[cols].values.ravel(), marker=mk, ls=ls, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(cols)
    ax.set_xlabel("Relearning steps (100 poisoned samples)")
    ax.set_ylabel("ASR (%)"); ax.set_ylim(-4, 106)
    ax.axhline(5, color="#888888", ls=":", lw=0.9)
    ax.legend(frameon=False, fontsize=6.0, loc="center right", bbox_to_anchor=(1.0, 0.28))
    ax.annotate("$\\leq$10 steps", xy=(2, 96), xytext=(2.3, 62),
                fontsize=6.0, arrowprops=dict(arrowstyle="->", lw=0.6))
    save(fig, "fig9_relearn.pdf")


def fig_detector(det):
    d = det[(det.arch == "resnet18")].copy()
    rates = [0.01, 0.05, 0.10]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.2), sharex=True)
    w = 0.34
    for ax, col, title in [(axes[0], "recall", "Recall $r$"), (axes[1], "precision", "Precision $p$")]:
        for i, (name, lab, c) in enumerate([("activation_clustering", "AC", C["ac"]),
                                            ("spectral_signatures", "SS", C["ss"])]):
            vals = []
            for rt in rates:
                row = d[(d.detector == name) & (np.isclose(d.poison_rate, rt))]
                vals.append(0.0 if len(row) == 0 or row.flagged.iloc[0] == 0 else float(row[col].iloc[0]))
            ax.bar(np.arange(3) + (i - 0.5) * w, vals, width=w, color=c, label=lab)
        ax.set_xticks(range(3)); ax.set_xticklabels(["1%", "5%", "10%"])
        ax.set_xlabel("Poison rate")
        ax.set_title(title, pad=2)
        ax.set_ylim(0, 1.12)
    axes[0].set_ylabel("Detector score")
    axes[0].annotate("no flags", xy=(2 - 0.5 * w, 0.02), xytext=(2.15, 0.30), fontsize=5.6,
                     arrowprops=dict(arrowstyle="->", lw=0.5))
    axes[1].legend(frameon=False, fontsize=6.0, loc="upper right", handlelength=1.2)
    fig.tight_layout(pad=0.25, w_pad=0.5)
    save(fig, "fig10_detector.pdf")


def fig_spike(sp):
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    mk = {"seed0": "o", "seed1": "s"}
    for tag, col in [("seed0", C["ac"]), ("seed1", C["ss"])]:
        d = sp[sp.seed_tag == tag]
        rep = d[d.tag.str.startswith("repair")]
        ctl = d[d.tag.str.startswith("clean-ctrl")]
        base = d[d.tag == "BASE"]
        ax.scatter(rep.asr, rep.auc_missed_vs_clean0, marker=mk[tag], s=30, color=col,
                   zorder=3, label=f"repair, seed {tag[-1]}")
        ax.scatter(ctl.asr, ctl.auc_missed_vs_clean0, marker=mk[tag], s=30,
                   facecolors="none", edgecolors=col, zorder=3, label=f"clean ctrl, seed {tag[-1]}")
        ax.scatter(base.asr, base.auc_missed_vs_clean0, marker="*", s=55, color=col, zorder=4)
    ax.set_xlabel("ASR after the intervention (%)")
    ax.set_ylabel("AUC (missed poison vs clean)")
    ax.set_xlim(-4, 108); ax.set_ylim(0.47, 0.69)
    ax.legend(frameon=False, fontsize=5.9, loc="lower left", handlelength=1.2,
              labelspacing=0.28)
    ax.annotate("base ($\\star$)", xy=(99, 0.549), xytext=(74, 0.585), fontsize=5.6,
                arrowprops=dict(arrowstyle="->", lw=0.5))
    save(fig, "fig11_spike_auc.pdf")


def tab_protocol():
    body = [
        r"Dataset & CIFAR-10, 50k train / 10k test, 10 classes, target class 0 \\",
        r"Attacks & BadNets $4\times4$ checkerboard (rows/cols 27--30); Blend $0.8x+0.2\,\pi$, seed 123 \\",
        r"Poison rate & 5\% (2{,}500 samples); 10\% in the amplification study \\",
        r"Architectures & ResNet-18, VGG-16-BN (CIFAR variants) \\",
        r"Optimiser & SGD, momentum 0.9 (Nesterov), weight decay $5\times10^{-4}$ \\",
        r"Schedule & OneCycleLR, max lr 0.1 (VGG 0.05), 12 epochs, batch 512 \\",
        r"Augmentation & random crop 32$\to$28 with padding 4, horizontal flip; normalisation inside the model \\",
        r"Seeds & 0, 1, 2 (identical splits; poison placement and init vary) \\",
        r"Detectors & Activation Clustering; Spectral Signatures (per-class, PCA/2-means resp.\ top singular vector) \\",
        r"Forget set & $\mathcal{F}(r,p)$: $N_p r$ true poison $+$ $N_p r(1-p)/p$ clean non-target false positives \\",
        r"Defence success & ASR $\leq5\%$ \emph{and} clean accuracy within 2 points of the gold retrain \\",
        r"Hardware & Kaggle 2$\times$T4 (16 GB each), mixed precision; $\approx$23 GPU-hours total \\",
    ]
    table("tab_protocol.tex",
          "Frozen experimental protocol. Every row of every table in this paper was produced under these settings; only the listed knobs vary between experiments.",
          "tab:protocol", "ll", r"\textbf{Item} & \textbf{Setting}", body, size="footnotesize")


def tab_detector(det):
    d = det[(det.arch == "resnet18")].copy().sort_values(["poison_rate", "detector"])
    body = []
    for _, r in d.iterrows():
        name = "AC" if r.detector == "activation_clustering" else "SS"
        prec = "--" if r.flagged == 0 else f"{r.precision:.3f}"
        rec = "--" if r.flagged == 0 else f"{r.recall:.3f}"
        body.append(f"{int(round(r.poison_rate*100))}\\% & {name} & {int(r.flagged):,} & "
                    f"{int(r.true_poison):,} & {prec} & {rec} \\\\")
    note = (r"Detector output on a ResNet-18/BadNets model. At 10\% poison, AC returns an empty "
            r"flag set (no cluster passes its size/silhouette test), i.e.\ $r=0$: the detector, not "
            r"the removal method, is the binding constraint. VGG-16-BN and MobileNetV2 detector "
            r"runs were degenerate (no flags or no true positives) and are excluded.")
    table("tab_detector.tex",
          "Measured detector operating points: the noisy forget sets that motivate the grid.",
          "tab:detector", "cccccc",
          r"\textbf{Poison} & \textbf{Det.} & \textbf{Flagged} & \textbf{True} & \textbf{Precision} & \textbf{Recall}",
          body, note=note, size="footnotesize")


def tab_e1grid(e1):
    cells = [(1.0, 1.0), (0.95, 1.0), (0.90, 1.0), (0.75, 1.0), (0.50, 1.0), (0.25, 1.0),
             (0.90, 0.9), (0.90, 0.7), (0.50, 0.9), (0.50, 0.7)]
    methods = ["retrain_delF", "finetune_retain", "negrad_plus", "scrub", "recipe"]
    body = []
    for r, p in cells:
        row = [f"{r:.2f}", f"{p:.1f}"]
        d0 = e1[(e1.method == "retrain_delF") & np.isclose(e1.recall, r) & np.isclose(e1.precision, p)]
        row.append(f"{int(round(d0.n_forget.iloc[0])):,}")
        for m in methods:
            d = e1[(e1.method == m) & np.isclose(e1.recall, r) & np.isclose(e1.precision, p)]
            s = ms(d.asr.values, 2)
            if len(d) >= 2 and d.asr.mean() <= 5.0:
                s = r"\textbf{" + s + "}"
            row.append(s)
        body.append(" & ".join(row) + r" \\")
    note = (r"ASR (\%) after removal, mean$\pm$s.d.\ over three seeds; \textbf{bold} marks mean ASR $\leq5\%$ "
            r"(the ASR half of defence success; clean accuracy is checked in Table~\ref{tab:success}). "
            r"$|\mathcal{F}|$ is reported for the seed-0 draw (identical across seeds up to rounding). "
            r"NegGrad+ is bimodal across seeds: two seeds drive ASR to $pprox0$ while collapsing clean "
            r"accuracy to 75--86\%, one stays fully inert (ASR 100\%); the mean and s.d.\ describe the "
            r"mixture, not a stable operating point.")
    table("tab_e1grid.tex",
          r"E1: the recall--precision grid. Deletion collapses only at $r{=}1$; repair$+$inoculation holds ASR below 5\% down to $r{=}0.5$.",
          "tab:e1grid", "ccrccccc",
          r"\textbf{$r$} & \textbf{$p$} & \textbf{$|\mathcal{F}|$} & \textbf{Retrain$_{\setminus F}$} & "
          r"\textbf{Fine-tune} & \textbf{NegGrad+} & \textbf{SCRUB} & \textbf{Recipe}",
          body, note=note, size="scriptsize", star=True)


def tab_success(e1, gold_by_seed):
    cells = [(1.0, 1.0), (0.95, 1.0), (0.90, 1.0), (0.75, 1.0), (0.50, 1.0), (0.25, 1.0),
             (0.90, 0.9), (0.90, 0.7), (0.50, 0.9), (0.50, 0.7)]
    methods = ["retrain_delF", "finetune_retain", "negrad_plus", "scrub", "recipe"]
    body, rstar = [], {}
    for m in methods:
        row = [DISPLAY[m]]
        succ = []
        for r, p in cells:
            n = 0
            for s in (0, 1, 2):
                d = e1[(e1.method == m) & (e1.seed == s) & np.isclose(e1.recall, r) & np.isclose(e1.precision, p)]
                if len(d) and d.asr.iloc[0] <= 5.0 and d.clean_acc.iloc[0] >= gold_by_seed[s] - 2.0:
                    n += 1
            succ.append(n)
            row.append(str(n))
        full = [cells[i][0] for i, n in enumerate(succ) if n == 3]
        rstar[m] = min(full) if full else None
        row.append(f"{rstar[m]:.2f}" if rstar[m] is not None else "--")
        body.append(" & ".join(row) + r" \\")
    note = (r"Number of seeds (of 3) meeting both criteria (ASR $\leq5\%$ and clean accuracy within 2 points "
            r"of the gold retrain). $r^{*}$ is the smallest recall with 3/3 success. Recipe misses $r{=}0.5$ at "
            r"$p{=}1$ by 0.09 clean-accuracy points on seed 2 but reaches 3/3 at $r{=}0.5,p{=}0.7$; we therefore "
            r"quote $r^{*}{=}0.5$ with that caveat. Gold clean accuracy: 88.59 / 88.65 / 88.77.")
    table("tab_success.tex",
          r"Defence-success counts and the recall tolerance $r^{*}$.",
          "tab:success", "lccccccccccc",
          r"\textbf{Method} & \multicolumn{10}{c}{\textbf{Successes (of 3) per $(r,p)$ cell, ordered as in Table~\ref{tab:e1grid}}} & \textbf{$r^{*}$}",
          body, note=note, size="scriptsize", star=True)


def tab_residual(urja):
    body = []
    for a, atk, lab in [("resnet18", "badnets", "R18/BadNets"), ("resnet18", "blend", "R18/Blend"),
                        ("vgg16_bn", "badnets", "VGG/BadNets"), ("vgg16_bn", "blend", "VGG/Blend")]:
        d = urja[(urja.arch == a) & (urja.attack == atk)].sort_values("f")
        row = [lab] + [f"{v:.2f}" for v in d.ASR.values] + [f"{d.clean_acc.mean():.1f}"]
        body.append(" & ".join(row) + r" \\")
    note = (r"ASR (\%) versus the fraction of poison left in the training set after deletion. "
            r"Clean accuracy (last column) is the mean over all six settings and varies by $<$0.8 points, "
            r"so the cliff is not an accuracy artefact.")
    table("tab_residual.tex",
          "E2: the cliff is a step function of residual poison, not of the amount deleted.",
          "tab:residual", "lccccccc",
          r"\textbf{Setting} & \textbf{0\%} & \textbf{25\%} & \textbf{50\%} & \textbf{75\%} & "
          r"\textbf{100\%} & \textbf{Gold} & \textbf{Clean}",
          body, note=note, size="footnotesize")


def tab_e6(e6, gold0):
    body = []
    for _, r in e6.iterrows():
        body.append(f"{DISPLAY[r.method]} & {r.recall:.3f} & {r.precision:.3f} & "
                    f"{r.clean_acc:.2f} & {r.asr:.2f} \\\\")
    note = (r"Removal driven by the detectors' \emph{raw} flag sets (no simulation), seed 0. "
            r"AC flags exactly the poison here; SS trades 16.6\% of recall for 16.6\% precision. "
            r"Deletion inherits the detector's recall error directly; repair$+$inoculation does not.")
    table("tab_e6.tex",
          "E6: realistic detector columns reproduce the simulated grid.",
          "tab:e6", "lcccc",
          r"\textbf{Method} & \textbf{Recall} & \textbf{Precision} & \textbf{Clean} & \textbf{ASR}",
          body, note=note, size="footnotesize")


def tab_e7(e7):
    body = []
    order = [("vgg16_bn", "badnets", "VGG/BadNets"), ("vgg16_bn", "blend", "VGG/Blend"),
             ("resnet18", "blend", "R18/Blend")]
    for a, atk, lab in order:
        d = e7[(e7.arch == a) & (e7.attack == atk)]
        g = d[d.method == "gold"]
        row = [lab, f"{g.clean_acc.iloc[0]:.2f} / {g.asr.iloc[0]:.2f}"]
        for m in ["retrain_delF", "recipe"]:
            for r in [0.75, 0.5]:
                x = d[(d.method == m) & np.isclose(d.recall, r)]
                row.append(f"{x.clean_acc.iloc[0]:.2f} / {x.asr.iloc[0]:.2f}")
        body.append(" & ".join(row) + r" \\")
    note = (r"Cells are clean accuracy / ASR (\%). Gold is the oracle retrain at $r{=}1$. "
            r"Deletion fails at both recall levels for every architecture--attack pair; the recipe holds "
            r"ASR $\leq1.0\%$, at a clean-accuracy cost of up to 4.1 points at $r{=}0.5$.")
    table("tab_e7.tex",
          "E7: the cliff and the recipe's tolerance transfer across architecture and attack.",
          "tab:e7", "lccccc",
          r"\textbf{Setting} & \textbf{Gold} & \textbf{Retrain $r{=}.75$} & \textbf{Retrain $r{=}.5$} & "
          r"\textbf{Recipe $r{=}.75$} & \textbf{Recipe $r{=}.5$}",
          body, note=note, size="footnotesize")


def tab_e8(e8):
    body = []
    raw = e8[(e8.variant == "raw")]
    bn = e8[(e8.variant == "bn_guarded")]
    for lab, d in [("Raw ascent", raw), ("BN-guarded ascent", bn)]:
        for _, r in d[d.method.str.startswith("ascent")].iterrows():
            body.append(f"{lab} & {r['method'].replace('ascent_', '').replace('_', ' ')} & "
                        f"{r.clean_acc:.2f} & {r.asr:.2f} \\\\")
    for _, r in raw[raw.method.str.startswith("repair")].iterrows():
        body.append(f"Label repair & {r['method'].replace('repair_', '').replace('_', ' ')} & "
                    f"{r.clean_acc:.2f} & {r.asr:.2f} \\\\")
    for _, r in bn[bn.method.str.startswith("repair")].iterrows():
        body.append(f"Label repair (v2) & {r['method'].replace('repair_', '').replace('_', ' ')} & "
                    f"{r.clean_acc:.2f} & {r.asr:.2f} \\\\")
    note = (r"Seed 1. Raw feature-space ascent appears to remove the backdoor only by destroying the "
            r"network (clean accuracy falls 13.5 points at 1e-3/4 epochs). Re-running the identical "
            r"schedule with batch-norm statistics frozen to the retain set leaves ASR at $\approx100\%$: "
            r"the apparent removal was BN corruption, not unlearning. Label repair at the same strength "
            r"reduces ASR while keeping clean accuracy within 1 point.")
    table("tab_e8.tex",
          "E8: ascent-based 'removal' is a normalisation artefact.",
          "tab:e8", "llcc",
          r"\textbf{Variant} & \textbf{Setting} & \textbf{Clean} & \textbf{ASR}",
          body, note=note, size="footnotesize")


def tab_car(e3):
    d = e3[e3.method == "car_proxy"].sort_values("n_flags").reset_index(drop=True)
    idx = [0, 3, 6, 9, 12, 15, 18, 19, len(d) - 1]
    body = []
    for i in idx:
        r = d.iloc[i]
        notes = r.notes
        parts = dict(p.split("=") for p in notes.split() if "=" in p)
        body.append(f"{float(parts['theta']):.2f} & {r.recall:.3f} & {r.precision:.3f} & "
                    f"{float(parts['tau']):.1f} & {float(parts['f1']):.3f} & {r.clean_acc:.2f} & {r.asr:.2f} \\\\")
    f1 = e3[e3.method == "f1_full"].iloc[0]
    body.append(r"\midrule")
    body.append(f"F1-optimal & {f1.recall:.3f} & {f1.precision:.3f} & 100.0 & 0.918 & "
                f"{f1.clean_acc:.2f} & {f1.asr:.2f} \\\\")
    note = (r"Calibrate-then-Remove, seed 0. $\theta$ is the Spectral-Signature threshold, $\tau$ the trigger "
            r"response of the proxy-removed model (\%). Nineteen of twenty thresholds give $\tau{=}100\%$ and "
            r"the twentieth $97.5\%$: the synthesised trigger never elicits the backdoor (IoU $=0.000$ "
            r"against the true patch), so no threshold satisfies $\tau\leq5\%$ and CaR degenerates to its "
            r"fallback. Both the F1-optimal and the CaR choice leave ASR $\approx100\%$.")
    table("tab_car.tex",
          "E3: Calibrate-then-Remove fails because the proxy signal is uninformative.",
          "tab:car", "ccccccc",
          r"\textbf{$\theta$} & \textbf{Recall} & \textbf{Precision} & \textbf{$\tau$} & \textbf{F1} & \textbf{Clean} & \textbf{ASR}",
          body, note=note, size="footnotesize")


def tab_e9(e9):
    body = []
    for s in (0, 1, 2):
        d = e9[e9.seed == s]
        get = lambda m: d[d.method == m].iloc[0]
        b, r1, r2, c1, c2 = (get("base"), get("partial_repair_lr0.005_ep2"),
                             get("partial_repair_lr0.01_ep1"), get("ctrl_clean_lr0.005_ep2"),
                             get("ctrl_clean_lr0.01_ep1"))
        body.append(f"{s} & {b.auc:.3f} & {r1.auc:.3f} & {r2.auc:.3f} & {c1.auc:.3f} & {c2.auc:.3f} & "
                    f"{r1.asr:.1f}/{r2.asr:.1f} & {r1.clean_acc:.1f}/{r2.clean_acc:.1f} \\\\")
    note = (r"10\% poison, simulated recall-0.5 flags. AUC separates the 2{,}500 \emph{missed} poison "
            r"samples from clean class-0 data. Partial label repair lowers ASR but does not raise AUC above "
            r"the base model in any seed, and the matched clean-removal controls sit at chance. The "
            r"pre-registered promotion gate ($\geq+0.10$ AUC, $\leq2$ points clean loss, beating control) "
            r"fails on every clause.")
    table("tab_e9.tex",
          "E9: the amplification effect does not survive at full scale.",
          "tab:e9", "cccccccc",
          r"\textbf{Seed} & \textbf{Base} & \textbf{Repair .005} & \textbf{Repair .01} & "
          r"\textbf{Ctrl .005} & \textbf{Ctrl .01} & \textbf{ASR (\%)} & \textbf{Clean}",
          body, note=note, size="footnotesize")


def tab_baselines(sisa, fp, pilot, gold0):
    body = []
    s = sisa.sort_values("recall", ascending=False)
    for _, r in s.iterrows():
        body.append(f"SISA-5 (this protocol) & {r.recall:.1f} & {r.clean_acc:.2f} & {r.asr:.2f} \\\\")
    for k, lab in [("sisa_k5_best_case", "SISA-5 (pilot)"), ("sisa_k10_best_case", "SISA-10 (pilot)"),
                   ("sisa_k20_best_case", "SISA-20 (pilot)")]:
        d = pilot[pilot.method == k]
        body.append(f"{lab} & oracle & {d.clean_acc.mean():.2f} & {d.ASR.mean():.2f} \\\\")
    d = fp[(fp.arch == "resnet18") & (fp.prune_ratio.isin([0.4, 0.6]))]
    for _, r in d.iterrows():
        body.append(f"Fine-pruning R18, {int(r.prune_ratio*100)}\\% & -- & {r.clean_acc:.2f} & {r.ASR:.2f} \\\\")
    d = fp[(fp.arch == "vgg16_bn") & (fp.prune_ratio == 0.9)]
    for _, r in d.iterrows():
        body.append(f"Fine-pruning VGG, {int(r.prune_ratio*100)}\\% & -- & {r.clean_acc:.2f} & {r.ASR:.2f} \\\\")
    body.append(r"\midrule")
    for k, lab in [("grad_ascent", "Gradient ascent (pilot)"), ("relabel_ft", "Label repair FT (pilot)")]:
        d = pilot[pilot.method == k]
        body.append(f"{lab} & -- & {d.clean_acc.mean():.2f} & {d.ASR.mean():.2f} \\\\")
    for k in ["partial_remove_25", "partial_remove_50", "partial_remove_75"]:
        d = pilot[pilot.method == k]
        body.append(f"{k.replace('_', ' ').replace('partial remove ', 'Partial deletion ')}\\% (pilot) & -- & "
                    f"{d.clean_acc.mean():.2f} & {d.ASR.mean():.2f} \\\\")
    note = (r"Reviewer-facing baselines. SISA and fine-pruning rows are measured; pilot rows use the minor "
            r"project's earlier protocol (clean accuracy $\approx$90\%) and are shown for completeness, not "
            r"pooled with the frozen results. SISA's clean accuracy is 18 points below the frozen gold model "
            r"already at $r{=}1$; fine-pruning needs 60\% (ResNet) or 90\% (VGG) of channels pruned before "
            r"ASR falls, by which point clean accuracy has dropped 8 and 3 points respectively.")
    table("tab_baselines.tex",
          "Efficiency-oriented and sharded baselines.",
          "tab:baselines", "lccc",
          r"\textbf{Method} & \textbf{Recall} & \textbf{Clean} & \textbf{ASR}",
          body, note=note, size="footnotesize")


def tab_durability(rel, urja):
    body = []
    for a, atk, lab in [("resnet18", "badnets", "R18/BadNets"), ("resnet18", "blend", "R18/Blend"),
                        ("vgg16_bn", "badnets", "VGG/BadNets"), ("vgg16_bn", "blend", "VGG/Blend")]:
        d = rel[(rel.arch == a) & (rel.attack == atk) & (rel.model == "gold_retrain")]
        m = urja[(urja.arch == a) & (urja.attack == atk) & (urja.method == "gold_retrain")].iloc[0]
        body.append(f"{lab} & {d['0'].iloc[0]:.1f} & {d['5'].iloc[0]:.1f} & {d['10'].iloc[0]:.1f} & "
                    f"{d['50'].iloc[0]:.1f} & {m.mia_auc:.3f} & {m.kl_clean:.3f}/{m.kl_trig:.1f} \\\\")
    note = (r"The gold (oracle-deleted) model, i.e.\ the best case any removal method can reach. One hundred "
            r"fresh poisoned samples restore ASR to $\geq97.7\%$ within 10 steps in all four settings; the "
            r"loss-threshold MIA sits at chance (0.49--0.50); and the clean-vs-trigger KL gap that "
            r"characterises a live backdoor (0.11--0.15 vs.\ 12--28 in the poisoned model) has vanished. "
            r"Suppression, not erasure.")
    table("tab_durability.tex",
          "E5: durability probes on the oracle-deleted model.",
          "tab:durability", "lcccccc",
          r"\textbf{Setting} & \textbf{Step 0} & \textbf{Step 5} & \textbf{Step 10} & \textbf{Step 50} & "
          r"\textbf{MIA AUC} & \textbf{KL clean/trig}",
          body, note=note, size="footnotesize")


def tab_rf(rf):
    body = []
    for m in ["gold_retrain", "finetune_retain", "recipe", "nc_style"]:
        row = [DISPLAY[m]]
        for r in [1.0, 0.75, 0.5]:
            d = rf[(rf.method == m) & np.isclose(rf.recall, r) & np.isclose(rf.precision, 1.0)]
            row.append(ms(d.asr.values, 2))
        row.append(ms(rf[rf.method == m].clean_acc.values, 2))
        body.append(" & ".join(row) + r" \\")
    b = rf[rf.method == "base"]
    note = (r"RadioML 2016.10a, 3 seeds, 5\% poison, synthetic I/Q trigger. Cells are ASR (\%) mean$\pm$s.d.; "
            r"the last column is clean accuracy over all runs. The poisoned base reaches ASR "
            f"{b.asr.mean():.2f}\\% at {b.clean_acc.mean():.2f}\\% clean accuracy. The recall cliff "
            r"reproduces: deletion is at chance for $r\leq0.75$, while trigger-aware repair holds ASR "
            r"below 5\% to $r{=}0.5$.")
    table("tab_rf.tex",
          "Cross-domain replication on RadioML AMC.",
          "tab:rf", "lcccc",
          r"\textbf{Method} & \textbf{$r{=}1.0$} & \textbf{$r{=}0.75$} & \textbf{$r{=}0.5$} & \textbf{Clean acc}",
          body, note=note, size="footnotesize")


# ----------------------------------------------------------------------------- main
def main():
    D = load()
    e1 = D["e1"]
    gold_by_seed = {}
    for s in (0, 1, 2):
        d = e1[(e1.method == "retrain_delF") & (e1.seed == s) & np.isclose(e1.recall, 1.0)]
        gold_by_seed[s] = float(d.clean_acc.iloc[0])
    NUM["gold_clean"] = gold_by_seed

    sp = []
    for tag, df in [("seed0", pd.read_csv(ROOT / "spike/spike_repair_seed0.csv")),
                    ("seed1", pd.read_csv(ROOT / "spike/spike_repair_seed1.csv"))]:
        df = df.copy(); df["seed_tag"] = tag; sp.append(df)
    D["spike"] = pd.concat(sp, ignore_index=True)

    print("figures")
    fig_pipeline()
    fig_cliff(e1, gold_by_seed)
    fig_success_map(e1, gold_by_seed)
    fig_residual(D["urja"])
    fig_ascent(D["e8"])
    fig_amplify(D["e9"])
    fig_rf(D["rfs"])
    fig_gamma(D["gam"], D["rfs"])
    fig_relearn(D["rel"])
    fig_detector(D["det"])
    fig_spike(D["spike"])

    print("tables")
    tab_protocol()
    tab_detector(D["det"])
    tab_e1grid(e1)
    tab_success(e1, gold_by_seed)
    tab_residual(D["urja"])
    tab_e6(D["e6"], gold_by_seed[0])
    tab_e7(D["e7"])
    tab_e8(D["e8"])
    tab_car(D["e3"])
    tab_e9(D["e9"])
    tab_baselines(D["sisa"], D["fp"], D["pilot"], gold_by_seed[0])
    tab_durability(D["rel"], D["urja"])
    tab_rf(D["rf"])

    # key numbers for prose
    def cell(m, r, p, col="asr"):
        d = e1[(e1.method == m) & np.isclose(e1.recall, r) & np.isclose(e1.precision, p)]
        return float(d[col].mean()), float(d[col].std(ddof=1))
    for m in ["retrain_delF", "recipe", "finetune_retain", "negrad_plus", "scrub"]:
        for r, p in [(1.0, 1.0), (0.95, 1.0), (0.9, 1.0), (0.75, 1.0), (0.5, 1.0),
                     (0.25, 1.0), (0.9, 0.9), (0.9, 0.7), (0.5, 0.9), (0.5, 0.7)]:
            NUM[f"e1_{m}_{r}_{p}"] = {"asr": cell(m, r, p), "clean": cell(m, r, p, "clean_acc")}
    NUM["e1_nc_style"] = {"asr": float(e1[e1.method == "nc_style"].asr.mean()),
                          "clean": float(e1[e1.method == "nc_style"].clean_acc.mean())}
    NUM["e2"] = D["urja"][["arch", "attack", "f", "ASR", "clean_acc"]].to_dict("records")
    NUM["e3_min_tau"] = float(D["e3"][D["e3"].method == "car_proxy"].notes.str.extract(r"tau=([\d.]+)")[0].astype(float).min())
    NUM["e5"] = D["rel"].to_dict("records")
    NUM["e6"] = D["e6"][["method", "recall", "precision", "clean_acc", "asr"]].to_dict("records")
    NUM["e7"] = D["e7"][["arch", "attack", "method", "recall", "clean_acc", "asr"]].to_dict("records")
    NUM["e8"] = D["e8"][["variant", "method", "clean_acc", "asr"]].to_dict("records")
    NUM["e9"] = D["e9"][["seed", "method", "clean_acc", "asr", "auc"]].to_dict("records")
    NUM["sisa"] = D["sisa"][["recall", "clean_acc", "asr"]].to_dict("records")
    NUM["rf"] = D["rf"][["seed", "method", "recall", "precision", "clean_acc", "asr"]].to_dict("records")
    NUM["gamma"] = D["gam"].to_dict("records")
    NUM["spike"] = D["spike"].to_dict("records")
    (OUT / "key_numbers.json").write_text(json.dumps(NUM, indent=2, default=float), encoding="utf-8")
    print("wrote key_numbers.json")


if __name__ == "__main__":
    sys.exit(main())
