"""Additional paper figures: pipeline schematics and data views.

Reads the same frozen CSVs as build_paper_assets.py; does not touch the
existing figures, tables or key_numbers.json.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
import build_paper_assets as B

C = B.C
D = B.DISPLAY


def box(ax, x, y, w, h, title, sub="", fc="#ffffff", fs=6.4, sfs=5.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.10,rounding_size=0.35",
                                fc=fc, ec="#000000", lw=0.7))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=fs, weight="bold")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=sfs, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, weight="bold")


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=7, lw=0.8, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ schematics
def fig_detection_pipeline():
    fig, ax = plt.subplots(figsize=(7.1, 1.75))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    ys = [19.5, 3.0]; h = 10.0
    xs = [0.5, 20.0, 39.5, 59.0, 78.5]
    ws = [17.5, 17.5, 17.5, 17.5, 20.5]
    ac = [("Features $Z$", "penultimate layer"),
          ("PCA ($d{=}10$)", "per class, $k{=}2$ means"),
          ("Small cluster", "size $<0.35$"),
          ("Silhouette", "$>0.10$"),
          ("Flags $\\hat{\\mathcal{F}}_{\\mathrm{AC}}$", "all classes pooled")]
    ss = [("Features $Z$", "penultimate layer"),
          ("Centre", "class mean removed"),
          ("Top singular", "vector $v_c$"),
          ("Score", "$(z_i^{\\top}v_c)^2$"),
          ("Flags $\\hat{\\mathcal{F}}_{\\mathrm{SS}}$", "top $1.5N_p$")]
    for y, row, lab, col in [(ys[0], ac, "AC", "#dbe9f4"),
                             (ys[1], ss, "SS", "#fbe3d0")]:
        ax.text(0.0, y + h + 1.6, lab, fontsize=7.0, weight="bold",
                ha="left", va="bottom")
        for i, (t, s) in enumerate(row):
            box(ax, xs[i], y, ws[i], h, t, s, fc=col)
            if i < 4:
                arrow(ax, xs[i] + ws[i], y + h / 2,
                      xs[i + 1], y + h / 2)
    ax.text(99.5, ys[0] + h + 1.6, "empty flag set at 10% poison",
            fontsize=5.8, ha="right", va="bottom", color="#8b1a1a", style="italic")
    ax.text(99.5, ys[1] + h + 1.6, "33.8% recall at 1% poison",
            fontsize=5.8, ha="right", va="bottom", color="#8b1a1a", style="italic")
    ax.text(50, 0.1, "Both detectors read the same penultimate features; either can supply the forget set.",
            fontsize=5.6, ha="center", va="bottom", style="italic")
    B.save(fig, "fig12_detection_pipeline.pdf")


def fig_removal_families():
    fig, ax = plt.subplots(figsize=(7.1, 2.25))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    h = 8.2
    xs = [14.0, 34.0, 54.0, 76.0]
    ws = [17.5, 17.5, 19.0, 22.0]
    rows = [
        ("Deletion", [("Random init", "new network"), ("$\\mathcal{D}\\setminus\\mathcal{F}$", "retained set"),
                      ("12 epochs", "full schedule"), ("Model", "no poison seen")], "#eeeeee"),
        ("Fine-tune", [("Poisoned $f$", "starting point"), ("$\\mathcal{D}\\setminus\\mathcal{F}$", "retained set"),
                       ("3 epochs", "lr 0.01"), ("Model", "poison kept")], "#fbe3d0"),
        ("Ascent", [("Poisoned $f$", "starting point"), ("$\\mathcal{F}$ + $\\mathcal{D}\\setminus\\mathcal{F}$", "forget + retain"),
                    ("3 epochs", "ascent / KL"), ("Model", "apparent suppression")], "#fbe3d0"),
        ("Repair+inoc.", [("Poisoned $f$", "starting point"), ("$\\mathcal{F}$ + trigger pool", "5k stamped"),
                          ("relabel + mix", "60/20/20"), ("Model", "poison re-labelled")], "#dbe9f4"),
    ]
    for j, (name, cells, col) in enumerate(rows):
        y = 40 - j * 10.0
        ax.text(0.0, y + h / 2, name, fontsize=6.4, weight="bold", ha="left", va="center")
        for i, (t, s) in enumerate(cells):
            fc = col if i < 3 else "#ffffff"
            box(ax, xs[i], y, ws[i], h, t, s, fc=fc, fs=6.0, sfs=5.2)
            if i < 3:
                arrow(ax, xs[i] + ws[i], y + h / 2, xs[i + 1], y + h / 2)
    ax.text(50, 0.0, "Only the last family uses the trigger; the other three depend on the forget set being complete.",
            fontsize=5.6, ha="center", va="bottom", style="italic")
    B.save(fig, "fig13_removal_families.pdf")


def fig_recipe_pipeline():
    fig, ax = plt.subplots(figsize=(3.45, 4.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 120); ax.axis("off")

    box(ax, 18, 107, 64, 10, "Poisoned model $f$", "penultimate features $Z$", fc="#eeeeee")
    box(ax, 18, 93, 64, 10, "Detector", "flags $\\hat{\\mathcal{F}}$ at recall $r$, precision $p$", fc="#dbe9f4")
    arrow(ax, 50, 107, 50, 103)
    arrow(ax, 50, 93, 50, 89)

    ax.plot([50, 50], [89, 85], color="#333333", lw=0.8)
    ax.plot([25, 75], [85, 85], color="#333333", lw=0.8)
    arrow(ax, 25, 85, 25, 81)
    arrow(ax, 75, 85, 75, 81)

    box(ax, 1, 60, 47, 20, "Flagged: repair", "kNN $k=10$ over unflagged\nnon-target data; keep sample\nwith pseudo-label",
        fc="#dbe9f4", fs=6.2, sfs=5.4)
    box(ax, 52, 60, 47, 20, "Unflagged: inoculate", "5,000 samples the model\ndoes not confidently call target;\ntrigger-stamp, keep true label",
        fc="#dbe9f4", fs=6.2, sfs=5.4)

    ax.plot([25, 25], [60, 56], color="#333333", lw=0.8)
    ax.plot([75, 75], [60, 56], color="#333333", lw=0.8)
    ax.plot([25, 75], [56, 56], color="#333333", lw=0.8)
    arrow(ax, 50, 56, 50, 52)

    box(ax, 10, 40, 80, 11, "Batch: 60% retained / 20% repaired / 20% inoculated", "", fc="#ffffff", fs=6.0)
    arrow(ax, 50, 40, 50, 36)
    box(ax, 10, 25, 80, 10, "Fine-tune 3 epochs, lr 0.01", "", fc="#ffffff", fs=6.2)
    arrow(ax, 50, 25, 50, 21)
    box(ax, 14, 10, 72, 10, "Removed model", "trigger no longer predicts target", fc="#e2f0d9", fs=6.4)
    ax.text(50, 3.0, "Tolerates missed poison: the trigger is presented\nwith correct labels, not deleted.",
            fontsize=5.6, ha="center", va="bottom", style="italic")
    B.save(fig, "fig14_recipe_pipeline.pdf")


def fig_code_arch():
    fig, ax = plt.subplots(figsize=(7.1, 1.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 24); ax.axis("off")
    y, h = 7.5, 10.5
    xs = [0.3, 17.2, 34.1, 51.0, 67.9, 84.8]
    ws = [15.2, 15.2, 15.2, 15.2, 15.2, 14.9]
    cells = [
        ("Frozen core", "core_kaggle.py\ndata, attacks, removal"),
        ("Drivers A--D", "Kaggle $2\\times$T4\nSai, Sarang,\nDivyansh, Urja"),
        ("Run logs", "runs_*.csv\none row per run"),
        ("Merge", "master combined\nCSV (all blocks)"),
        ("Asset builder", "build_paper\n_assets.py"),
        ("Paper", "figures, tables\nmain.tex $\\to$ PDF"),
    ]
    for i, (t, s) in enumerate(cells):
        box(ax, xs[i], y, ws[i], h, t, s, fc="#dbe9f4" if i in (0, 1) else "#ffffff",
            fs=6.0, sfs=5.0)
        if i < 5:
            arrow(ax, xs[i] + ws[i], y + h / 2, xs[i + 1], y + h / 2)
    ax.text(50, 1.8, "The core is never edited between drivers: seeds, epochs, learning rates and batch size are fixed; "
            "every number in the paper is read from a CSV, none is transcribed.",
            fontsize=5.6, ha="center", va="bottom", style="italic")
    B.save(fig, "fig15_code_arch.pdf")


# ------------------------------------------------------------------ data views
def fig_grid_views(e1):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.15))
    ax = axes[0, 0]
    cells_r = [1.0, 0.95, 0.9, 0.75, 0.5, 0.25]
    methods = [("retrain_delF", "retrain", "-"), ("finetune_retain", "finetune", "--"),
               ("negrad_plus", "negrad", ":"), ("scrub", "scrub", "-."),
               ("recipe", "recipe", "-")]
    gold = e1[(e1.method == "retrain_delF") & np.isclose(e1.recall, 1.0)].clean_acc.mean()
    ax.axhspan(-2, 2, color="#0072B2", alpha=0.10, lw=0)
    ax.axhline(0, color="#888888", lw=0.6)
    for m, key, ls in methods:
        xs, ys = [], []
        for r in cells_r:
            d = e1[(e1.method == m) & np.isclose(e1.recall, r) & np.isclose(e1.precision, 1.0)]
            if len(d):
                xs.append(r); ys.append(d.clean_acc.mean() - gold)
        ax.plot(xs, ys, marker="o", ms=2.4, ls=ls, lw=1.5 if m == "recipe" else 1.1,
                color=C[key], label=D[m])
    ax.set_xlim(0.2, 1.03); ax.set_ylim(-25, 3)
    ax.set_xticks([0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    ax.set_xticklabels(["0.25", "0.5", "0.75", "0.9", "0.95", "1"])
    ax.set_xlabel("Recall $r$ (precision $p=1$)")
    ax.set_ylabel("Clean acc. minus gold (pts)")
    ax.set_title("(a) utility cost across recall", fontsize=7.2, pad=2)
    ax.text(0.27, -14, "2-pt budget", fontsize=5.4, color="#005a8f", ha="left", va="top")

    for i, (ax, r) in enumerate([(axes[0, 1], 0.9), (axes[1, 0], 0.5)]):
        for m, key, ls in methods:
            ps, ys = [], []
            for p in [0.7, 0.9, 1.0]:
                d = e1[(e1.method == m) & np.isclose(e1.recall, r) & np.isclose(e1.precision, p)]
                if len(d):
                    ps.append(p); ys.append(d.asr.mean())
            ax.plot(ps, ys, marker="o", ms=2.4, ls=ls, lw=1.5 if m == "recipe" else 1.1,
                    color=C[key], label=D[m])
        ax.set_xlim(0.66, 1.03); ax.set_ylim(-4, 106)
        ax.set_xticks([0.7, 0.9, 1.0]); ax.set_xticklabels(["0.7", "0.9", "1.0"])
        ax.set_xlabel(f"Precision $p$ (recall $r={r:g}$)")
        ax.set_ylabel("ASR (%)")
        ax.set_title(f"({'bc'[i]}) ASR across precision", fontsize=7.2, pad=2)
        ax.axhspan(0, 5, color="#0072B2", alpha=0.10, lw=0)

    ax = axes[1, 1]
    for m, key, ls in methods:
        ps, ys = [], []
        for p in [0.7, 0.9, 1.0]:
            d = e1[(e1.method == m) & np.isclose(e1.recall, 0.5) & np.isclose(e1.precision, p)]
            if len(d):
                ps.append(p); ys.append(d.clean_acc.mean() - gold)
        ax.plot(ps, ys, marker="o", ms=2.4, ls=ls, lw=1.5 if m == "recipe" else 1.1,
                color=C[key], label=D[m])
    ax.axhspan(-2, 2, color="#0072B2", alpha=0.10, lw=0)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xlim(0.66, 1.03); ax.set_ylim(-33, 3)
    ax.set_xticks([0.7, 0.9, 1.0]); ax.set_xticklabels(["0.7", "0.9", "1.0"])
    ax.set_xlabel("Precision $p$ (recall $r{=}0.5$)")
    ax.set_ylabel("Clean acc. minus gold (pts)")
    ax.set_title("(d) utility cost across precision", fontsize=7.2, pad=2)

    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, fontsize=6.0, loc="lower center",
               bbox_to_anchor=(0.5, -0.015), handlelength=1.4, columnspacing=1.1)
    fig.tight_layout(pad=0.25, w_pad=0.7, h_pad=0.55, rect=(0, 0.055, 1, 1))
    B.save(fig, "fig16_grid_views.pdf")


def fig_e7_transfer(e7):
    settings = [("vgg16_bn", "badnets", "VGG/BadNets"),
                ("vgg16_bn", "blend", "VGG/Blend"),
                ("resnet18", "blend", "R18/Blend")]
    series = [("retrain_delF", 0.75, "Deletion $r{=}0.75$", "s", C["retrain"]),
              ("retrain_delF", 0.5, "Deletion $r{=}0.5$", "D", C["retrain"]),
              ("recipe", 0.75, "Recipe $r{=}0.75$", "o", C["recipe"]),
              ("recipe", 0.5, "Recipe $r{=}0.5$", "^", C["nc"])]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.55))
    x = np.arange(len(settings)); w = 0.19
    ax = axes[0]
    for i, (m, r, lab, mk, col) in enumerate(series):
        vals = []
        for a, atk, _ in settings:
            d = e7[(e7.arch == a) & (e7.attack == atk) & (e7.method == m) & np.isclose(e7.recall, r)]
            vals.append(float(d.asr.iloc[0]))
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=col, hatch="///" if m == "retrain_delF" else None,
               edgecolor="white" if m == "recipe" else "none", linewidth=0.3, label=lab)
    golds = []
    for a, atk, _ in settings:
        d = e7[(e7.arch == a) & (e7.attack == atk) & (e7.method == "gold")]
        golds.append(float(d.asr.iloc[0]))
    ax.scatter(x, golds, marker="_", s=60, color="#000000", zorder=5, label="Gold")
    ax.set_yscale("log"); ax.set_ylim(0.2, 300)
    ax.set_xticks(x); ax.set_xticklabels([s[2] for s in settings], fontsize=5.8, rotation=12)
    ax.set_ylabel("ASR (%, log scale)")
    ax.set_title("(a) attack success", fontsize=7.2, pad=2)
    ax.axhline(5, color="#888888", lw=0.7, ls="--")

    ax = axes[1]
    for i, (m, r, lab, mk, col) in enumerate(series):
        vals = []
        for a, atk, _ in settings:
            g = e7[(e7.arch == a) & (e7.attack == atk) & (e7.method == "gold")].clean_acc.iloc[0]
            d = e7[(e7.arch == a) & (e7.attack == atk) & (e7.method == m) & np.isclose(e7.recall, r)]
            vals.append(float(d.clean_acc.iloc[0] - g))
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=col,
               hatch="///" if m == "retrain_delF" else None,
               edgecolor="white" if m == "recipe" else "none", linewidth=0.3)
    ax.axhspan(-2, 2, color="#0072B2", alpha=0.10, lw=0)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([s[2] for s in settings], fontsize=5.8, rotation=12)
    ax.set_ylabel("Clean acc. minus gold (pts)")
    ax.set_ylim(-5.2, 1.6)
    ax.set_title("(b) utility cost", fontsize=7.2, pad=2)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, fontsize=5.8, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), handlelength=1.2, columnspacing=0.8,
               labelspacing=0.25)
    fig.tight_layout(pad=0.25, w_pad=0.6, rect=(0, 0.12, 1, 1))
    B.save(fig, "fig17_e7_transfer.pdf")


def fig_durability(urja):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.35))
    styles = {("resnet18", "badnets"): ("o", "-", "#0072B2"),
              ("resnet18", "blend"): ("s", "--", "#0072B2"),
              ("vgg16_bn", "badnets"): ("^", "-", "#D55E00"),
              ("vgg16_bn", "blend"): ("D", "--", "#D55E00")}
    for ax, col, lab, ylim in [(axes[0], "mia_auc", "Membership-inference AUC", (0.48, 0.52)),
                               (axes[1], "kl_trig", "KL divergence, triggered", (0, 31))]:
        for (a, atk), (mk, ls, c) in styles.items():
            d = urja[(urja.arch == a) & (urja.attack == atk)].sort_values("f")
            ax.plot(d.f * 100, d[col], marker=mk, ms=2.8, ls=ls, color=c,
                    label=f"{'R18' if a=='resnet18' else 'VGG'}/{'BadNets' if atk=='badnets' else 'Blend'}")
        ax.set_xlabel("Poison removed (%)")
        ax.set_ylabel(lab)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(*ylim)
    axes[0].axhline(0.5, color="#888888", ls=":", lw=0.8)
    axes[0].set_title("(a) membership probe", fontsize=7.2, pad=2)
    axes[1].set_title("(b) output divergence", fontsize=7.2, pad=2)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, fontsize=5.6, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), handlelength=1.3, columnspacing=0.8,
               labelspacing=0.22)
    fig.tight_layout(pad=0.25, w_pad=0.7, rect=(0, 0.16, 1, 1))
    B.save(fig, "fig18_durability_probes.pdf")


def fig_rf_heatmap(rfs):
    recalls = [1.0, 0.75, 0.5]
    methods = [("base", "Base"), ("gold_retrain", "Deletion"), ("finetune_retain", "Fine-tune"),
               ("recipe", "Recipe"), ("nc_style", "Inoc.-only")]
    snrs = sorted(rfs.snr.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), sharey=True)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#e6e6e6")
    im = None
    for ax, r in zip(axes, recalls):
        M = np.zeros((len(methods), len(snrs)))
        for i, (m, _) in enumerate(methods):
            d = rfs[(rfs.method == m) & np.isclose(rfs.recall, r) & np.isclose(rfs.precision, 1.0)]
            g = d.groupby("snr").asr.mean()
            M[i, :] = [g.get(s, np.nan) for s in snrs]
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0, vmax=100,
                       extent=[min(snrs) - 1, max(snrs) + 1, len(methods) - 0.5, -0.5])
        for i in range(len(methods)):
            if np.isnan(M[i]).all():
                ax.text(np.mean(snrs), i, "not run", ha="center", va="center",
                        fontsize=5.6, color="#666666", style="italic")
        ax.set_title(f"recall $r={r:g}$", fontsize=7.0, pad=2)
        ax.set_xticks([-20, -10, 0, 10, 18])
        ax.set_xlabel("SNR (dB)")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([lab for _, lab in methods], fontsize=6.2)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.012, label="ASR (%)")
    B.save(fig, "fig19_rf_heatmap.pdf")


def fig_rf_clean(rfs):
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    ax.axvspan(-21, -10, color="#eeeeee", lw=0)
    ax.plot([], [], " ", alpha=0)
    curves = [("base", "Poisoned base", "#999999", 2.4, "-", None),
              ("gold_retrain", "Gold retrain", "#000000", 1.1, "--", None),
              ("recipe", "Recipe", C["recipe"], 1.3, "-", "o")]
    for m, lab, col, lw, ls, mk in curves:
        d = rfs[(rfs.method == m) & np.isclose(rfs.recall, 1.0) & np.isclose(rfs.precision, 1.0)]
        g = d.groupby("snr").clean_acc.mean()
        ax.plot(g.index, g.values, color=col, lw=lw, ls=ls,
                marker=mk, ms=2.2, label=lab, zorder=3 if m == "recipe" else 2)
    ax.axhline(9.1, color="#888888", ls=":", lw=0.8)
    ax.text(2, 12.0, "chance (9.1%)", fontsize=5.4, color="#666666")
    ax.axvline(-6, color="#aaaaaa", ls="--", lw=0.7)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Clean accuracy (%)")
    ax.set_xlim(-21, 19); ax.set_ylim(0, 95)
    ax.set_xticks([-20, -10, 0, 10, 18])
    ax.legend(frameon=False, fontsize=5.9, loc="upper left")
    B.save(fig, "fig20_rf_clean.pdf")


def main():
    d = B.load()
    print("schematics")
    fig_detection_pipeline()
    fig_removal_families()
    fig_recipe_pipeline()
    fig_code_arch()
    print("data")
    fig_grid_views(d["e1"])
    fig_e7_transfer(d["e7"])
    fig_durability(d["urja"])
    fig_rf_heatmap(d["rfs"])
    fig_rf_clean(d["rfs"])


if __name__ == "__main__":
    main()
