# -*- coding: utf-8 -*-
"""
生成期末论文用的英文图（arXiv 风格，300 dpi）：
  fig1_arch.png      蒸馏→剪枝→量化→演示 流水线结构图
  fig2_curves.png    训练损失曲线（教师 / 从零 / 蒸馏）
  fig3_ablation.png  (a) T×α 热力图  (b) 学生步数扫描
  fig4_tradeoff.png  压缩阶梯：体积 vs 留出集损失
运行： python paper_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_distill")
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Fig. 1  结构图
# ------------------------------------------------------------
def fig_arch():
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.set_xlim(0, 12.9)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#3b6ea5", fs=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
                                    linewidth=0.9, facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.9, color="#444444", shrinkA=0, shrinkB=0))

    box(0.05, 2.60, 1.90, 1.05, r"Corpus" + "\n" + r"$5.5\times10^{4}$ chars" + "\n" + r"$|V|=322$",
        fc="#f4f4f4", ec="#777777", fs=6.9)
    box(2.20, 2.55, 2.05, 1.10, "Teacher GPT\n2 layers, $d$=160\n0.74 M params")
    box(2.20, 0.75, 2.05, 1.10, "Student GPT\n2 layers, $d$=64\n0.15 M params",
        fc="#eaf6ee", ec="#3f8f5c")
    box(4.75, 1.50, 2.85, 1.25,
        "KD loss\n" + r"$(1-\alpha)\,\mathrm{CE}+\alpha T^{2}\,\mathrm{KL}$", fs=7.4)
    box(8.15, 2.55, 1.75, 1.10, "Magnitude\npruning")
    box(8.15, 0.60, 1.75, 1.10, "Dynamic\nint8")
    box(10.55, 1.50, 1.55, 1.10, "Interactive\ndemo", fc="#fdf3e7", ec="#c98a3c")

    arrow(1.95, 3.12, 2.20, 3.10)
    arrow(1.35, 2.60, 2.20, 1.45)
    arrow(4.25, 2.95, 4.75, 2.45)
    arrow(4.25, 1.45, 4.75, 1.95)
    arrow(7.60, 2.65, 8.15, 3.00)
    arrow(7.60, 1.60, 8.15, 1.20)
    arrow(9.90, 3.00, 10.55, 2.35)
    arrow(9.90, 1.20, 10.55, 1.95)

    ax.text(3.22, 3.88, "knowledge transfer", fontsize=6.6, ha="center", color="#3b6ea5")
    ax.text(0.05, 0.08, "Every stage runs on CPU (24 threads): no GPU and no external pretrained weights.",
            fontsize=6.6, color="#555555")
    fig.savefig(os.path.join(FIGS, "fig1_arch.png"))
    plt.close(fig)


# ------------------------------------------------------------
# Fig. 2  训练曲线
# ------------------------------------------------------------
def fig_curves():
    res = load("results_all.json")
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for key, label, color, ls in (("teacher", "Teacher (0.74 M)", "#1f4e79", "-"),
                                  ("scratch", "Student from scratch (0.15 M)", "#c0504d", "--"),
                                  ("kd", "Student + KD (0.15 M)", "#2e7d32", "-")):
        curve = res[key]["curve"]
        if not curve:
            continue
        xs = list(range(len(curve)))
        ys = curve
        ax.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=1.2)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Training loss")
    ax.set_title("Distillation dynamics on a 50 k-character corpus")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIGS, "fig2_curves.png"))
    plt.close(fig)


# ------------------------------------------------------------
# Fig. 3  消融：(a) T×α 热力图  (b) 步数扫描
# ------------------------------------------------------------
def fig_ablation():
    abl = load("results_ablate.json")["ablation"]
    steps = load("results_steps.json")["steps"]
    Ts = sorted({r["T"] for r in abl})
    As = sorted({r["alpha"] for r in abl})
    grid = np.array([[next(r["val_loss"] for r in abl if r["T"] == T and r["alpha"] == a)
                      for a in As] for T in Ts])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.45})
    ax = axes[0]
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(As)), ["%.1f" % a for a in As])
    ax.set_yticks(range(len(Ts)), ["%d" % T for T in Ts])
    ax.set_xlabel(r"Soft-label weight $\alpha$")
    ax.set_ylabel(r"Temperature $T$")
    ax.set_title("(a) Held-out loss over $T\\times\\alpha$")
    mid = grid.min() + 0.5 * (grid.max() - grid.min())
    for i in range(len(Ts)):
        for j in range(len(As)):
            ax.text(j, i, "%.3f" % grid[i, j], ha="center", va="center", fontsize=6.5,
                    color="white" if grid[i, j] > mid else "black")
    bi, bj = np.unravel_index(int(grid.argmin()), grid.shape)
    ax.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False, edgecolor="#d62728", lw=1.8))
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)

    ax = axes[1]
    xs = [s["iters"] for s in steps]
    ax.plot(xs, [s["scratch"] for s in steps], "o-", color="#c0504d", label="From scratch", lw=1.2, ms=4)
    ax.plot(xs, [s["kd"] for s in steps], "s-", color="#2e7d32", label="KD student", lw=1.2, ms=4)
    for s in steps:
        ax.annotate("%+.1f%%" % ((s["scratch"] - s["kd"]) / s["scratch"] * 100),
                    (s["iters"], max(s["scratch"], s["kd"])), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.5, color="#333333")
    ax.set_xlabel("Student training iterations")
    ax.set_ylabel("Held-out loss")
    ax.set_title("(b) Gain vs. training budget")
    ax.margins(x=0.07, y=0.18)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIGS, "fig3_ablation.png"))
    plt.close(fig)


# ------------------------------------------------------------
# Fig. 4  压缩阶梯
# ------------------------------------------------------------
def fig_tradeoff():
    pipe = load("results_pipeline.json")
    base = {r["tag"]: r for r in pipe["baselines"]}
    width = pipe["width"]
    pruned = [r for r in pipe["pruned"] if r.get("ratio", 0) > 0]
    pipe_q = pipe["pipeline"][0]

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ws = [w["nbytes"] / 1024 / 1024 for w in width]
    ls = [w["loss"] for w in width]
    ax.plot(ws, ls, "o-", color="#2e7d32", lw=1.2, ms=5, label="KD student, width sweep")
    for w in width:
        ax.annotate("$d$=%d" % w["width"], (w["nbytes"] / 1024 / 1024, w["loss"]),
                    textcoords="offset points", xytext=(5, -10), fontsize=6.5, color="#2e7d32")
    ax.scatter([base["teacher"]["nbytes"] / 1024 / 1024], [base["teacher"]["loss"]],
               marker="*", s=110, color="#1f4e79", zorder=3, label="Teacher (0.74 M)")
    ax.scatter([base["student_kd"]["nbytes"] / 1024 / 1024], [base["student_kd"]["loss"]],
               marker="D", s=30, color="#8c564b", zorder=3, label="Canonical KD student")
    px = [p["nbytes"] / 1024 / 1024 for p in pruned]
    py = [p["loss"] for p in pruned]
    ax.plot(px, py, "^--", color="#9467bd", lw=1.0, ms=4,
            label="Magnitude pruning (20-75%)")
    ax.scatter([pipe_q["nbytes"] / 1024 / 1024], [pipe_q["loss"]], marker="X", s=60,
               color="#d62728", zorder=4, label="Pruned 20% + int8 (0.31 MB)")
    ax.set_xscale("log")
    ax.set_xlim(0.22, 3.4)
    ax.set_xticks([0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: "%g" % v))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Checkpoint size on disk (MB, log scale)")
    ax.set_ylabel("Held-out cross-entropy")
    ax.set_title("Accuracy--size trade-off of the compression ladder")
    ax.grid(alpha=0.3, linewidth=0.5, which="both")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(os.path.join(FIGS, "fig4_tradeoff.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_arch()
    fig_curves()
    fig_ablation()
    fig_tradeoff()
    print("figures saved to", FIGS)
