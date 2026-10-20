"""Summary figure from simulation results.

Usage (from the repository root):
    python scripts/make_figure.py results/exploratory_2026-10-20
Reads res_exp1/exp2/exp3/exp3b/exp5.json from the given directory and writes
figure_summary.png into the same directory.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})


def load(d, exp):
    r = json.load(open(os.path.join(d, f"res_{exp}.json")))
    return r["results"] if "results" in r else r


def main(d):
    r1, r2, r3, r3b, r5 = (load(d, e) for e in ("exp1", "exp2", "exp3", "exp3b", "exp5"))
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2), gridspec_kw=dict(width_ratios=[1.4, 1.2, 1]))

    # A: false positives
    ax = axs[0]
    cats = ["equal\nscale", "scale\n0.8", "scale\n0.6", "15% random\nresponders", "group-specific\nitems"]
    keys = ["equal scale", "lam 0.8", "lam 0.6", "15% random responders"]
    cnt = [r1[k]["pc"]["fp_rate_per_item"] for k in keys] + [r2["counts_fp_rate"]]
    nai = [r1[k]["pn"]["fp_rate_per_item"] for k in keys] + [np.nan]
    adj = [r1[k]["ps"]["fp_rate_per_item"] for k in keys] + [r2["model_fp_rate"]]
    x = np.arange(len(cats))
    w = 0.26
    for i, (vals, col, lab) in enumerate([(cnt, C1, "count scores (Welch test)"),
                                          (nai, C2, "logit, equal scale assumed"),
                                          (adj, C3, "logit, free scale per group")]):
        xs = x + (i - 1) * w
        ax.bar(xs, np.nan_to_num(vals), w - 0.03, color=col, label=lab, zorder=3)
        for xx, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(xx, v + 0.015, f"{v * 100:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0.05, color=INK2, lw=1, ls=(0, (3, 3)), zorder=2, label="nominal 5%")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("share of items with p < 0.05")
    ax.set_title("A  False positives in group comparisons\n(no true preference difference, N = 150 per group)",
                 loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # B: joint design vs. blocks
    ax = axs[1]
    labs = ["1 set\n14 items", "2 blocks\nof 7", "blocks\n+1 bridge", "blocks\n+2 bridges", "blocks\n+3 bridges"]
    tasks = [14, 14, 15, 16, 17]
    win = [r3["sd_within_joint"], r3["sd_within_blocks"]] + [r3b[f"bridge{n}"]["sd_within"] for n in (1, 2, 3)]
    crs = [r3["sd_cross_joint"], np.nan] + [r3b[f"bridge{n}"]["sd_cross"] for n in (1, 2, 3)]
    x = np.arange(len(labs))
    w = 0.36
    ax.bar(x - w / 2, win, w - 0.03, color=C1, label="within domain", zorder=3)
    ax.bar(x + w / 2, np.nan_to_num(crs), w - 0.03, color=C2, label="across domains", zorder=3)
    for xx, v in zip(x - w / 2, win):
        ax.text(xx, v + 0.003, f"{v:.3f}", ha="center", fontsize=6.8, color=INK2)
    for xx, v in zip(x + w / 2, crs):
        ax.text(xx, (0 if np.isnan(v) else v) + 0.003, "-" if np.isnan(v) else f"{v:.3f}",
                ha="center", fontsize=6.8, color=INK2)
    ax.set_xticks(x, [f"{l}\n({t} tasks)" for l, t in zip(labs, tasks)], fontsize=8)
    ax.set_ylabel("SD of estimated utility differences")
    ax.set_title("B  Precision: one joint set vs. two blocks\n(logit, N = 150, smaller = more precise; - = not estimable)",
                 loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, max(np.nanmax(win), np.nanmax(crs)) * 1.35)

    # C: polarising item
    ax = axs[2]
    p6 = np.array(r5["example_per6"])
    p5 = np.array(r5["example_per5"])
    vals = np.arange(-4, 5)                      # individual scores are multiples of 1/4
    c5 = [(np.round(p5 * 4) == v).sum() for v in vals]
    c6 = [(np.round(p6 * 4) == v).sum() for v in vals]
    xw = 0.38
    ax.bar(vals - xw / 2, c5, xw - 0.04, color=C1, label="homogeneous item (utility -0.17)", zorder=3)
    ax.bar(vals + xw / 2, c6, xw - 0.04, color=C2, label="polarising item (mean 0)", zorder=3)
    ax.set_xticks(vals, [f"{v / 4:.2f}" for v in vals], fontsize=8)
    ax.set_ylim(0, max(c5 + c6) * 1.3)
    ax.set_xlabel("individual count score (B - W) / 4")
    ax.set_ylabel("respondents")
    ax.set_title("C  A polarising item looks average in aggregate\n(individual utility +/-1.5; one sample, N = 150)",
                 loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center")

    for a in axs:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "Sequential best-worst logit, Gumbel errors, heterogeneity SD 0.5, items permuted per "
             "respondent. A: 200 replications, B: 300, C: one sample. Assumptions are placeholders.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(d, "figure_summary.png")
    fig.savefig(out, dpi=170)
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/latest")
